// @ts-nocheck

import { defineAgent, llm, voice } from '@livekit/agents';
import * as livekit from '@livekit/agents-plugin-livekit';
import * as silero from '@livekit/agents-plugin-silero';
import { BackgroundVoiceCancellation } from '@livekit/noise-cancellation-node';
import { z } from 'zod';

type ServiceType = 'collection' | 'delivery';
type DraftState = 'drafting' | 'confirming' | 'placing' | 'completed';

type DraftItem = {
  item_name: string;
  quantity: number;
  modifiers: string[];
  special_instructions?: string | null;
};

type OrderDraft = {
  state: DraftState;
  service_type: ServiceType | null;
  customer_name: string | null;
  pickup_time: string | null; // MUST be ASAP or HH:MM for your backend
  delivery_address: string | null;
  notes: string | null;
  items: DraftItem[];
  last_question?: string | null;
};

type UserData = {
  draft: OrderDraft;
  // You can optionally set menu here (comma-separated env or fetched later)
  menu: string[]; // canonical item names
};

const ORDERPILOT_ORDERS_URL = process.env.ORDERPILOT_ORDERS_URL || '';
const RESTAURANT_ID = process.env.RESTAURANT_ID || '';
const DEFAULT_SERVICE_TYPE = (process.env.DEFAULT_SERVICE_TYPE as ServiceType) || 'collection';

// Model strings (LiveKit Cloud inference-supported providers)
const STT_MODEL = process.env.STT_MODEL || 'deepgram/nova-2-phonecall:en';
const TTS_MODEL = process.env.TTS_MODEL || 'elevenlabs/english_v1'; // set to your actual available TTS model
const LLM_MODEL = process.env.LLM_MODEL || 'openai/gpt-4.1-mini';

// Optional: provide known menu items as CSV in env MENU_ITEMS="Pepperoni Pizza,Margherita Pizza,Fries,Coke"
const MENU_ITEMS = (process.env.MENU_ITEMS || '')
  .split(',')
  .map((s) => s.trim())
  .filter(Boolean);

function newDraft(): OrderDraft {
  return {
    state: 'drafting',
    service_type: DEFAULT_SERVICE_TYPE,
    customer_name: null,
    pickup_time: null,
    delivery_address: null,
    notes: null,
    items: [],
    last_question: null,
  };
}

function clampQty(qty: number | undefined | null): number {
  const n = Number(qty ?? 1);
  if (!Number.isFinite(n)) return 1;
  return Math.max(1, Math.min(50, Math.round(n)));
}

function normalizeText(s: string): string {
  return s.trim().replace(/\s+/g, ' ');
}

function isHHMM(s: string): boolean {
  return /^([01]\d|2[0-3]):[0-5]\d$/.test(s);
}

/**
 * Converts common user time phrases into ASAP or HH:MM (24h).
 * NOTE: This intentionally NEVER returns ISO timestamps because your backend rejects them.
 */
function normalizePickupTime(input: string): string | null {
  const raw = normalizeText(input).toLowerCase();
  if (!raw) return null;

  if (raw === 'asap' || raw.includes('asap') || raw.includes('now') || raw.includes('soonest')) {
    return 'ASAP';
  }

  // "in 20 minutes", "in 15 mins"
  const rel = raw.match(/in\s+(\d{1,3})\s*(min|mins|minute|minutes)\b/);
  if (rel) {
    const mins = Math.max(0, Math.min(180, parseInt(rel[1]!, 10)));
    const d = new Date();
    d.setMinutes(d.getMinutes() + mins);
    const hh = String(d.getHours()).padStart(2, '0');
    const mm = String(d.getMinutes()).padStart(2, '0');
    return `${hh}:${mm}`;
  }

  // "7pm", "7 pm", "7:30pm", "19:00"
  const hm =
    raw.match(/^(\d{1,2})\s*:\s*(\d{2})\s*(am|pm)?$/) ||
    raw.match(/^(\d{1,2})\s*(am|pm)$/);
  if (hm) {
    let h = parseInt(hm[1]!, 10);
    let m = 0;

    if (hm.length >= 4 && hm[2] && hm[3]) {
      // hh:mm am/pm
      m = parseInt(hm[2]!, 10);
      const ap = hm[3]!;
      if (ap === 'pm' && h < 12) h += 12;
      if (ap === 'am' && h === 12) h = 0;
    } else if (hm.length === 3 && hm[2]) {
      // hh am/pm
      const ap = hm[2]!;
      if (ap === 'pm' && h < 12) h += 12;
      if (ap === 'am' && h === 12) h = 0;
    } else if (hm.length >= 3 && hm[2] && !hm[3]) {
      // hh:mm 24h
      m = parseInt(hm[2]!, 10);
    }

    if (h >= 0 && h <= 23 && m >= 0 && m <= 59) {
      return `${String(h).padStart(2, '0')}:${String(m).padStart(2, '0')}`;
    }
  }

  // "half seven" / "half 7"
  const half = raw.match(/half\s+(\d{1,2})/);
  if (half) {
    let h = parseInt(half[1]!, 10);
    // assume "half seven" = 7:30 (UK)
    if (h >= 1 && h <= 12) {
      // heuristic: if caller says "half seven" on evening calls, likely PM; but we can’t guess reliably.
      // Keep it simple: treat 1-11 as 19:30 if current hour >= 12, else 07:30.
      const nowH = new Date().getHours();
      if (nowH >= 12 && h < 12) h += 12;
      return `${String(h).padStart(2, '0')}:30`;
    }
  }

  // If user accidentally gave ISO like "2025-12-28T20:00:00.000Z"
  // Convert to HH:MM (local server time).
  const iso = Date.parse(input);
  if (!Number.isNaN(iso)) {
    const d = new Date(iso);
    const hh = String(d.getHours()).padStart(2, '0');
    const mm = String(d.getMinutes()).padStart(2, '0');
    return `${hh}:${mm}`;
  }

  return null;
}

function canonicalizeItemName(name: string, menu: string[]): { ok: true; canonical: string } | { ok: false; suggestions: string[] } {
  const n = normalizeText(name);
  if (!n) return { ok: false, suggestions: [] };
  if (!menu.length) return { ok: true, canonical: n }; // if no menu configured, accept anything

  const lower = n.toLowerCase();
  const exact = menu.find((m) => m.toLowerCase() === lower);
  if (exact) return { ok: true, canonical: exact };

  // basic suggestion: contains or startswith
  const sug = menu
    .filter((m) => {
      const ml = m.toLowerCase();
      return ml.includes(lower) || lower.includes(ml) || ml.startsWith(lower) || lower.startsWith(ml);
    })
    .slice(0, 5);

  return { ok: false, suggestions: sug };
}

function summarizeDraft(d: OrderDraft): string {
  const items = d.items
    .map((it) => {
      const mods = it.modifiers?.length ? ` (${it.modifiers.join(', ')})` : '';
      const note = it.special_instructions ? ` [${it.special_instructions}]` : '';
      return `${it.quantity} x ${it.item_name}${mods}${note}`;
    })
    .join('; ');

  const count = d.items.reduce((a, b) => a + (b.quantity || 0), 0);
  const time = d.pickup_time || 'ASAP';
  const name = d.customer_name || '—';

  return `Name: ${name}. Time: ${time}. Items (${count}): ${items || '—'}.`;
}

function nextMissingQuestion(d: OrderDraft): string | null {
  if (!d.items.length) return 'What can I get you today?';
  if (!d.customer_name) return 'Can I get your name?';
  if (!d.pickup_time) return 'What time for collection? You can say ASAP or a time like 7pm.';
  return null;
}

async function postOrder(d: OrderDraft): Promise<{ ok: true; orderId?: string } | { ok: false; error: string }> {
  if (!ORDERPILOT_ORDERS_URL) return { ok: false, error: 'ORDERPILOT_ORDERS_URL is not set' };
  if (!RESTAURANT_ID) return { ok: false, error: 'RESTAURANT_ID is not set' };

  const pickup = d.pickup_time && isHHMM(d.pickup_time) ? d.pickup_time : d.pickup_time === 'ASAP' ? 'ASAP' : null;
  if (!pickup) {
    return { ok: false, error: 'Invalid time. Say "ASAP", "7pm", "in 20 minutes", or "19:00".' };
  }

  const payload = {
    restaurant_id: RESTAURANT_ID,
    customer_name: d.customer_name,
    customer_phone: null,
    pickup_time: pickup,
    service_type: d.service_type ?? DEFAULT_SERVICE_TYPE,
    notes: d.notes,
    items: d.items.map((it) => ({
      item_name: it.item_name,
      quantity: it.quantity,
      unit_price: null,
    })),
  };

  const controller = new AbortController();
  const t = setTimeout(() => controller.abort(), 8000);

  try {
    const res = await fetch(ORDERPILOT_ORDERS_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
      signal: controller.signal,
    });

    const text = await res.text();
    if (!res.ok) {
      return { ok: false, error: `Order create failed ${res.status}: ${text}` };
    }

    // best-effort parse
    let parsed: any = null;
    try {
      parsed = JSON.parse(text);
    } catch {
      parsed = null;
    }

    const orderId = parsed?.order_id || parsed?.id;
    return { ok: true, orderId: typeof orderId === 'string' ? orderId : undefined };
  } catch (e: any) {
    const msg = e?.name === 'AbortError' ? 'Timeout talking to OrderPilot backend' : String(e?.message || e);
    return { ok: false, error: msg };
  } finally {
    clearTimeout(t);
  }
}

const UpdateSchema = z.object({
  intent: z
    .enum(['order', 'add', 'remove', 'change', 'confirm', 'cancel', 'question', 'restart', 'unknown'])
    .describe('User intent'),
  service_type: z.enum(['collection', 'delivery']).optional(),
  customer_name: z.string().optional(),
  pickup_time: z.string().optional().describe('Raw time phrase like "7pm", "in 20 minutes", "ASAP"'),
  delivery_address: z.string().optional(),
  notes: z.string().optional(),
  items: z
    .array(
      z.object({
        name: z.string().describe('Menu item name as spoken'),
        quantity: z.number().optional().describe('Quantity, default 1'),
        modifiers: z.array(z.string()).optional().describe('Modifiers like "no mayo"'),
        special_instructions: z.string().optional().describe('Any notes'),
      })
    )
    .optional()
    .describe('Items mentioned in this turn'),
  remove_items: z.array(z.string()).optional().describe('Item names to remove'),
});

class OrderPilotAgent extends voice.Agent<UserData> {
  constructor() {
    super({
      instructions: `
You are OrderPilot, a professional UK takeaway phone assistant.

Goals:
- Fast, natural, minimal words.
- Take COLLECTION orders by default unless the user clearly wants delivery.
- The user can give info in any order; you must handle interruptions.
- You MUST always call the tool "update_order_draft" once per user turn BEFORE you respond, passing extracted fields.
- Only confirm ONCE with a short itemised summary and ask for "yes" to place it.
- After user confirms, call "create_order" exactly once. If it fails, retry once, then fallback politely.

Time rule (CRITICAL):
- pickup_time sent to backend MUST be "ASAP" or "HH:MM" (24h). Never send ISO.

Menu rule:
- If MENU_ITEMS is configured and the user asks for something not on the menu, do NOT accept it.
  Offer 2-3 closest suggestions and ask what they'd like instead.

FAQs:
- If asked about address/hours/payment/wait time and unknown, say you’re not sure and offer to take the order anyway.

Style:
- Short replies. One question at a time.
      `.trim(),

      // Provide tools for the LLM to use
      tools: {
        update_order_draft: llm.tool({
          description:
            'Extract structured intent and fields from the user message and update the current OrderDraft. Always call this once per user turn before responding.',
          parameters: UpdateSchema,
          execute: async (args, { ctx }) => {
            const ud = (ctx.session.userData ?? { draft: newDraft(), menu: MENU_ITEMS }) as UserData;
            ctx.session.userData = ud;

            const d = ud.draft;

            // Intent handlers
            if (args.intent === 'restart') {
              ud.draft = newDraft();
              return { ok: true, message: 'Order restarted.' };
            }
            if (args.intent === 'cancel') {
              ud.draft.state = 'completed';
              return { ok: true, message: 'Order cancelled.' };
            }

            if (args.service_type) d.service_type = args.service_type;
            if (args.customer_name) d.customer_name = normalizeText(args.customer_name);
            if (args.delivery_address) d.delivery_address = normalizeText(args.delivery_address);
            if (typeof args.notes === 'string') d.notes = normalizeText(args.notes);

            if (args.pickup_time) {
              const norm = normalizePickupTime(args.pickup_time);
              if (norm) d.pickup_time = norm;
            }

            // Remove items
            if (args.remove_items?.length) {
              const toRemove = args.remove_items.map((x) => x.toLowerCase());
              d.items = d.items.filter((it) => !toRemove.some((r) => it.item_name.toLowerCase().includes(r)));
            }

            // Add / change items
            if (args.items?.length) {
              for (const rawItem of args.items) {
                const name = normalizeText(rawItem.name);
                if (!name) continue;

                const match = canonicalizeItemName(name, ud.menu);
                if (!match.ok) {
                  // Signal to the LLM that this item is not on menu
                  return {
                    ok: false,
                    not_on_menu: name,
                    suggestions: match.suggestions,
                  };
                }

                const canonical = match.canonical;
                const qty = clampQty(rawItem.quantity ?? 1);
                const mods = (rawItem.modifiers ?? []).map(normalizeText).filter(Boolean);

                // If item exists, increase quantity (fast phone behavior)
                const existing = d.items.find((x) => x.item_name.toLowerCase() === canonical.toLowerCase());
                if (existing) {
                  existing.quantity = clampQty((existing.quantity ?? 1) + qty);
                  existing.modifiers = Array.from(new Set([...(existing.modifiers || []), ...mods])).filter(Boolean);
                  if (rawItem.special_instructions) {
                    existing.special_instructions = normalizeText(rawItem.special_instructions);
                  }
                } else {
                  d.items.push({
                    item_name: canonical,
                    quantity: qty,
                    modifiers: mods,
                    special_instructions: rawItem.special_instructions ? normalizeText(rawItem.special_instructions) : null,
                  });
                }
              }
            }

            // State transitions
            if (args.intent === 'confirm') {
              d.state = 'confirming';
            } else if (d.state === 'confirming' && args.intent !== 'confirm') {
              // user keeps talking; go back to drafting
              d.state = 'drafting';
            }

            const missing = nextMissingQuestion(d);
            d.last_question = missing;

            return {
              ok: true,
              draft: {
                state: d.state,
                service_type: d.service_type,
                customer_name: d.customer_name,
                pickup_time: d.pickup_time,
                delivery_address: d.delivery_address,
                notes: d.notes,
                items: d.items,
              },
              missing,
              summary: summarizeDraft(d),
            };
          },
        }),

        create_order: llm.tool({
          description:
            'Create the order in OrderPilot backend. Only call after the user says yes to the confirmation.',
          parameters: z.object({
            confirm: z.boolean().describe('Must be true only when user confirmed the order'),
          }),
          execute: async ({ confirm }, { ctx }) => {
            const ud = (ctx.session.userData ?? { draft: newDraft(), menu: MENU_ITEMS }) as UserData;
            ctx.session.userData = ud;

            if (!confirm) return { ok: false, error: 'Not confirmed' };

            const d = ud.draft;

            // Ensure minimum fields
            const missing = nextMissingQuestion(d);
            if (missing) {
              return { ok: false, error: `Missing info: ${missing}` };
            }

            d.state = 'placing';

            // 1st attempt
            let res = await postOrder(d);
            if (!res.ok) {
              // retry once
              res = await postOrder(d);
            }

            if (!res.ok) {
              d.state = 'drafting';
              return { ok: false, error: res.error };
            }

            d.state = 'completed';
            return { ok: true, orderId: res.orderId ?? null };
          },
        }),

        answer_faq: llm.tool({
          description:
            'Answer common restaurant FAQs (hours, address, payment, wait time). Use if asked. If unknown, say you’re not sure and offer to take the order.',
          parameters: z.object({
            question: z.string(),
          }),
          execute: async ({ question }) => {
            // Keep this simple + safe; customize later
            const q = question.toLowerCase();
            if (q.includes('address')) {
              const addr = process.env.RESTAURANT_ADDRESS;
              return addr
                ? { ok: true, answer: `We’re at ${addr}.` }
                : { ok: false, answer: `I’m not sure of the exact address — but I can take your order now.` };
            }
            if (q.includes('open') || q.includes('hours') || q.includes('closing')) {
              const hours = process.env.OPENING_HOURS;
              return hours
                ? { ok: true, answer: `Our opening hours are: ${hours}.` }
                : { ok: false, answer: `I’m not sure of today’s hours — but I can take your order now.` };
            }
            if (q.includes('card') || q.includes('cash') || q.includes('pay')) {
              const pay = process.env.PAYMENT_INFO;
              return pay
                ? { ok: true, answer: pay }
                : { ok: false, answer: `I’m not 100% sure — but I can take the order and the shop will confirm on pickup.` };
            }
            if (q.includes('how long') || q.includes('wait') || q.includes('ready')) {
              const wait = process.env.DEFAULT_WAIT_TIME;
              return wait
                ? { ok: true, answer: `Roughly ${wait}. Want it ASAP or a specific time?` }
                : { ok: false, answer: `It depends on how busy it is — do you want it ASAP or a specific time?` };
            }
            return { ok: false, answer: `I’m not sure — but I can take your order now.` };
          },
        }),
      },
    });
  }

  override async onEnter() {
    const session = this.session;
    if (!session.userData) {
      session.userData = { draft: newDraft(), menu: MENU_ITEMS } satisfies UserData;
    }

    // Short opening line
    await session.say('Hi, OrderPilot here. What can I get you today?', { allowInterruptions: true });
  }
}

export default defineAgent({
  entry: async (ctx) => {
    const vad = await silero.VAD.load();

    const session = new voice.AgentSession<UserData>({
      vad,
      stt: STT_MODEL,
      llm: LLM_MODEL,
      tts: TTS_MODEL,

      // Turn detection model (requires weights) — keep it on for better phone UX
      turnDetection: new livekit.turnDetector.EOUModel(),

      // Preemptive generation reduces perceived latency
      voiceOptions: {
        preemptiveGeneration: true,
      },

      // Make interruptions feel natural on phone
      allowInterruptions: true,

      // Seed userData
      userData: {
        draft: newDraft(),
        menu: MENU_ITEMS,
      },
    });

    await session.start({
      room: ctx.room,
      agent: new OrderPilotAgent(),
      inputOptions: {
        noiseCancellation: BackgroundVoiceCancellation(),
      },
    });

    // Optional observability (no TS hard dependency on enums)
    (session as any).on?.('close', () => {
      // noop
    });
  },
});
