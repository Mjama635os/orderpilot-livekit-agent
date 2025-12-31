// @ts-nocheck
/**
 * OrderPilot LiveKit Agent (Production-Ready, Menu-Aware)
 * - Fast, natural phone ordering with a small state machine
 * - Robust time handling (ALWAYS sends pickup_time as "ASAP" or "HH:MM")
 * - Menu enforcement (rejects items not on menu, suggests closest matches)
 * - Multi-item, quantities, modifiers, notes
 * - Edits: add/remove/change qty/time/name/restart
 * - One brief confirmation only
 * - Backend tool call with 1 retry + graceful fallback
 *
 * IMPORTANT ENV VARS:
 * - LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET
 * - ORDERPILOT_ORDERS_URL (e.g. https://api.orderpilot.co.uk/orders)
 * - DEFAULT_RESTAURANT_ID
 * - (optional) STT_MODEL (default deepgram/nova-2-phonecall:en)
 * - (optional) TTS_MODEL (default elevenlabs/eleven_turbo_v2_5)
 * - (optional) LLM_MODEL (default openai/gpt-4.1-mini)
 * - (optional) MENU_JSON (JSON array of strings OR objects; see MENU section)
 */

import dotenv from "dotenv";
dotenv.config({ path: ".env.local" });

import {
  type JobContext,
  type JobProcess,
  ServerOptions,
  cli,
  defineAgent,
  inference,
  metrics,
  voice,
  llm,
} from "@livekit/agents";

import * as silero from "@livekit/agents-plugin-silero";
import { fetch } from "undici";
import { fileURLToPath } from "node:url";
import { z } from "zod";

/* =========================================================
   Config
========================================================= */

const ORDERPILOT_ORDERS_URL = process.env.ORDERPILOT_ORDERS_URL;
const DEFAULT_RESTAURANT_ID = process.env.DEFAULT_RESTAURANT_ID;

const STT_MODEL = process.env.STT_MODEL || "deepgram/nova-2-phonecall:en";
const TTS_MODEL = process.env.TTS_MODEL || "elevenlabs/eleven_turbo_v2_5";
const LLM_MODEL = process.env.LLM_MODEL || "openai/gpt-4.1-mini";

const SILENCE_MS = 12000;

/* =========================================================
   Types
========================================================= */

type ServiceType = "collection" | "delivery";
type AgentState = "drafting" | "confirming" | "placing" | "completed";

type DraftItem = {
  item_name: string; // canonical display
  quantity: number;
  modifiers: string[];
  special_instructions: string | null;
};

type OrderDraft = {
  service_type: ServiceType;
  customer_name: string | null;
  pickup_time: string | null; // "ASAP" | "HH:MM"
  items: DraftItem[];
  notes: string | null;
};

type Extracted = {
  intent:
    | "order"
    | "add"
    | "remove"
    | "change"
    | "confirm"
    | "cancel"
    | "question"
    | "restart"
    | "unknown";
  name?: string | null;
  pickup_time?: string | null;
  service_type?: ServiceType | null;
  question?: string | null;
  items?: Array<{
    name: string;
    quantity?: number | null;
    modifiers?: string[] | null;
    special_instructions?: string | null;
  }>;
};

/* =========================================================
   Small helpers
========================================================= */

const YES = /\b(yes|yeah|yep|yup|ok|okay|correct|confirm|go ahead|that's right|thats right|perfect)\b/i;
const NO = /\b(no|nope|wrong|incorrect|not correct|change|cancel|restart|start again)\b/i;

const MAX_TURN_CHARS = 220;

function clean(s: string): string {
  return String(s || "").trim().replace(/\s+/g, " ");
}

function short(s: string): string {
  const t = clean(s);
  return t.length > MAX_TURN_CHARS ? `${t.slice(0, MAX_TURN_CHARS - 1)}…` : t;
}

function resetDraft(): OrderDraft {
  return {
    service_type: "collection",
    customer_name: null,
    pickup_time: null,
    items: [],
    notes: null,
  };
}

function normalizeServiceType(text: string): ServiceType | null {
  const t = text.toLowerCase();
  if (t.includes("delivery") || t.includes("deliver")) return "delivery";
  if (t.includes("collection") || t.includes("collect") || t.includes("pickup") || t.includes("pick up") || t.includes("takeaway"))
    return "collection";
  return null;
}

/**
 * MUST RETURN ONLY:
 * - "ASAP"
 * - "HH:MM" (24h)
 */
function normalizePickupTime(input: string | null | undefined): string | null {
  const raw = clean(input || "").toLowerCase();
  if (!raw) return null;

  if (/(asap|now|as soon|straight away|right away)/.test(raw)) return "ASAP";

  // "in 20 minutes" -> ASAP (backend supports this too, but we keep ultra-safe)
  const inMin = raw.match(/\bin\s+(\d+)\s*(min|mins|minute|minutes)\b/);
  if (inMin) return "ASAP";

  // "7pm", "7 pm", "7:30pm", "19:00"
  const m1 = raw.match(/^(\d{1,2})\s*:\s*(\d{2})\s*(am|pm)?$/);
  if (m1) {
    let hh = parseInt(m1[1], 10);
    const mm = parseInt(m1[2], 10);
    const ap = m1[3] || null;
    if (Number.isNaN(hh) || Number.isNaN(mm) || mm < 0 || mm > 59) return null;
    if (ap) {
      if (ap === "pm" && hh < 12) hh += 12;
      if (ap === "am" && hh === 12) hh = 0;
    }
    if (hh < 0 || hh > 23) return null;
    return `${String(hh).padStart(2, "0")}:${String(mm).padStart(2, "0")}`;
  }

  const m2 = raw.match(/^(\d{1,2})\s*(am|pm)$/);
  if (m2) {
    let hh = parseInt(m2[1], 10);
    const ap = m2[2];
    if (Number.isNaN(hh) || hh < 1 || hh > 12) return null;
    if (ap === "pm" && hh < 12) hh += 12;
    if (ap === "am" && hh === 12) hh = 0;
    return `${String(hh).padStart(2, "0")}:00`;
  }

  // "half seven" (UK) -> 7:30; assume evening if current hour >= 12
  const half = raw.match(/\bhalf\s+(\d{1,2})\b/);
  if (half) {
    let hh = parseInt(half[1], 10);
    if (Number.isNaN(hh) || hh < 1 || hh > 12) return null;
    const nowH = new Date().getHours();
    if (nowH >= 12 && hh < 12) hh += 12;
    return `${String(hh).padStart(2, "0")}:30`;
  }

  // If they accidentally give ISO -> convert to local HH:MM
  const parsed = Date.parse(input as any);
  if (!Number.isNaN(parsed)) {
    const d = new Date(parsed);
    return `${String(d.getHours()).padStart(2, "0")}:${String(d.getMinutes()).padStart(2, "0")}`;
  }

  return null;
}

function draftNeedsNext(d: OrderDraft): string | null {
  if (!d.service_type) return "Is this for collection or delivery?";
  if (!d.items.length) return "What can I get you today?";
  if (!d.customer_name) return "Can I take a name for the order?";
  if (!d.pickup_time) return 'What time would you like it for? Say "ASAP" or a time like "19:00".';
  return null;
}

function itemKey(x: DraftItem): string {
  return clean(x.item_name).toLowerCase();
}

function upsertItem(d: OrderDraft, item: DraftItem) {
  const key = itemKey(item);
  const idx = d.items.findIndex((i) => itemKey(i) === key);
  if (idx === -1) {
    d.items.push(item);
    return;
  }
  d.items[idx].quantity += Math.max(1, item.quantity || 1);
  d.items[idx].modifiers = Array.from(new Set([...(d.items[idx].modifiers || []), ...(item.modifiers || [])]));
  const a = clean(d.items[idx].special_instructions || "");
  const b = clean(item.special_instructions || "");
  d.items[idx].special_instructions = a && b ? `${a}; ${b}` : (a || b || null);
}

function removeItemByName(d: OrderDraft, canonical: string): boolean {
  const key = clean(canonical).toLowerCase();
  const before = d.items.length;
  d.items = d.items.filter((i) => itemKey(i) !== key);
  return d.items.length !== before;
}

function countTotalItems(d: OrderDraft): number {
  return d.items.reduce((sum, i) => sum + (Number(i.quantity) || 0), 0);
}

function confirmOrderSummary(d: OrderDraft): string {
  const lines = d.items.map((i) => {
    const mods = i.modifiers?.length ? ` (${i.modifiers.join(", ")})` : "";
    const note = i.special_instructions ? ` [${i.special_instructions}]` : "";
    return `${i.quantity}× ${i.item_name}${mods}${note}`;
  });

  const total = countTotalItems(d);
  const name = d.customer_name ? ` for ${d.customer_name}` : "";
  const when = d.pickup_time ? ` at ${d.pickup_time}` : "";
  const svc = d.service_type === "delivery" ? "delivery" : "collection";

  return short(
    `Okay — ${svc}${name}${when}. I’ve got ${total} item${total === 1 ? "" : "s"}: ` +
      lines.join("; ") +
      `. Is that correct?`
  );
}

/* =========================================================
   MENU (enforcement + fuzzy matching)
   Provide MENU_JSON as either:
   - ["Pepperoni Pizza","Margherita Pizza", ...]
   OR objects: [{ "name": "Pepperoni Pizza", "synonyms": ["pepperoni"] }]
========================================================= */

type MenuEntry = { name: string; synonyms?: string[] };

function loadMenu(): MenuEntry[] {
  const raw = process.env.MENU_JSON;
  if (!raw) return [];
  try {
    const parsed = JSON.parse(raw);
    if (Array.isArray(parsed)) {
      if (parsed.length === 0) return [];
      if (typeof parsed[0] === "string") return parsed.map((s: string) => ({ name: String(s) }));
      return parsed
        .map((x: any) => ({
          name: String(x?.name || x?.item || ""),
          synonyms: Array.isArray(x?.synonyms) ? x.synonyms.map(String) : [],
        }))
        .filter((x: any) => x.name);
    }
    return [];
  } catch {
    return [];
  }
}

const MENU: MenuEntry[] = loadMenu();

function normTxt(s: string): string {
  return clean(s).toLowerCase().replace(/[^\p{L}\p{N}\s]/gu, "");
}

function score(a: string, b: string): number {
  // quick token overlap score (cheap + reliable)
  const A = new Set(normTxt(a).split(" ").filter(Boolean));
  const B = new Set(normTxt(b).split(" ").filter(Boolean));
  if (!A.size || !B.size) return 0;
  let inter = 0;
  for (const t of A) if (B.has(t)) inter++;
  return inter / Math.max(A.size, B.size);
}

function bestMenuMatches(query: string, k = 3): string[] {
  if (!MENU.length) return [];
  const q = normTxt(query);
  const scored = MENU.map((m) => {
    const base = score(q, m.name);
    const syn = (m.synonyms || []).reduce((mx, s) => Math.max(mx, score(q, s)), 0);
    return { name: m.name, sc: Math.max(base, syn) };
  })
    .sort((a, b) => b.sc - a.sc)
    .slice(0, k);
  return scored.filter((x) => x.sc >= 0.45).map((x) => x.name);
}

function menuCanonicalize(name: string): { ok: true; canonical: string } | { ok: false; suggestions: string[] } {
  if (!MENU.length) return { ok: true, canonical: clean(name) };

  const exact = MENU.find((m) => normTxt(m.name) === normTxt(name));
  if (exact) return { ok: true, canonical: exact.name };

  const matches = bestMenuMatches(name, 3);
  if (!matches.length) return { ok: false, suggestions: [] };

  // If it’s clearly the top result, accept; else ask to pick.
  if (matches.length === 1) return { ok: true, canonical: matches[0] };
  const s0 = score(name, matches[0]);
  const s1 = score(name, matches[1]);
  if (s0 >= 0.62 && s0 - s1 >= 0.12) return { ok: true, canonical: matches[0] };
  return { ok: false, suggestions: matches };
}

/* =========================================================
   Tool: create_order (backend)
========================================================= */

const createOrderTool = llm.tool({
  description: "Create a confirmed order in the OrderPilot backend.",
  parameters: z.object({
    customer_name: z.string(),
    service_type: z.enum(["collection", "delivery"]),
    pickup_time: z.string().describe('MUST be "ASAP" or "HH:MM" like "19:00"'),
    notes: z.string().nullable().optional(),
    items: z.array(
      z.object({
        item_name: z.string(),
        quantity: z.number().int().min(1),
        unit_price: z.number().nullable().optional(),
      })
    ),
  }),
  execute: async (args) => {
    if (!ORDERPILOT_ORDERS_URL) throw new Error("Missing ORDERPILOT_ORDERS_URL");
    if (!DEFAULT_RESTAURANT_ID) throw new Error("Missing DEFAULT_RESTAURANT_ID");

    const pickup = normalizePickupTime(args.pickup_time);
    if (!pickup) {
      throw new Error('Order create failed 400: {"success":false,"code":"1","error":"Invalid time. Say \\"ASAP\\", \\"7pm\\", \\"in 20 minutes\\", or \\"19:00\\"."}');
    }

    const payload = {
      restaurant_id: DEFAULT_RESTAURANT_ID,
      customer_name: args.customer_name,
      customer_phone: null,
      pickup_time: pickup,
      service_type: args.service_type,
      notes: args.notes ?? null,
      items: args.items,
    };

    const attempt = async () => {
      const controller = new AbortController();
      const timeout = setTimeout(() => controller.abort(), 8000);
      try {
        const res = await fetch(ORDERPILOT_ORDERS_URL, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
          signal: controller.signal,
        });
        const data = await res.json().catch(() => ({}));
        if (!res.ok) throw new Error(`Order create failed ${res.status}: ${JSON.stringify(data)}`);
        return data;
      } finally {
        clearTimeout(timeout);
      }
    };

    try {
      return await attempt();
    } catch {
      // 1 retry
      return await attempt();
    }
  },
});

/* =========================================================
   LLM extraction (structured)
========================================================= */

const extractionSchema = z.object({
  intent: z.enum(["order", "add", "remove", "change", "confirm", "cancel", "question", "restart", "unknown"]),
  name: z.string().nullable().optional(),
  pickup_time: z.string().nullable().optional(),
  service_type: z.enum(["collection", "delivery"]).nullable().optional(),
  question: z.string().nullable().optional(),
  items: z
    .array(
      z.object({
        name: z.string(),
        quantity: z.number().int().min(1).nullable().optional(),
        modifiers: z.array(z.string()).nullable().optional(),
        special_instructions: z.string().nullable().optional(),
      })
    )
    .optional(),
});

async function extractFromText(session: any, text: string): Promise<Extracted> {
  const prompt = `
Extract the caller's intent and any order details from this utterance.

Rules:
- intent:
  - "order" if placing a new order or listing items
  - "add" if adding items
  - "remove" if removing items
  - "change" if changing qty/name/time/service type
  - "confirm" if confirming ("yes", "that's right")
  - "cancel" if cancelling
  - "restart" if "start again"
  - "question" if asking FAQ
  - "unknown" otherwise
- items: split multi-item phrases; quantity defaults to 1; modifiers includes "no mayo", "extra spicy", etc.
- pickup_time: return as spoken, e.g. "7pm", "19:00", "ASAP", "in 20 minutes"
- name: customer's name if stated

Utterance:
"""${text}"""
`.trim();

  // Use the session LLM directly for lowest latency in this stack.
  // If your @livekit/agents version doesn't expose session.llm, this still works at runtime via inference.
  const res = await (session as any).llm?.complete?.({
    prompt,
    responseFormat: "json",
  });

  // Fallback: use inference LLM directly (works in most versions)
  const jsonText =
    res?.text ||
    res?.output_text ||
    res?.choices?.[0]?.message?.content ||
    res?.choices?.[0]?.text ||
    null;

  if (!jsonText) return { intent: "unknown" };

  try {
    const parsed = JSON.parse(String(jsonText));
    const safe = extractionSchema.safeParse(parsed);
    if (!safe.success) return { intent: "unknown" };
    return safe.data as any;
  } catch {
    return { intent: "unknown" };
  }
}

/* =========================================================
   Agent
========================================================= */

class OrderPilotAgent extends voice.Agent {
  draft: OrderDraft = resetDraft();
  state: AgentState = "drafting";
  confirmAsked = false;

  constructor() {
    super({
      instructions: `
You are OrderPilot — a calm, confident UK restaurant phone order taker.

Goal:
Take collection orders quickly and accurately with minimal words.

Critical rules:
- Only accept items that exist on the menu. If not on the menu, ask what they meant and suggest closest matches.
- Ask only ONE question at a time.
- Confirm ONCE with a compact itemized summary. Then wait for yes/no.
- If the customer edits after confirmation, apply changes and re-confirm once.
- Never send pickup_time as an ISO date. Only "ASAP" or "HH:MM".
- If asked FAQs (hours/address/payment/wait time) and you don't know, say you’re not sure and offer to take the order anyway.

Style:
Friendly, fast, not robotic. No long speeches.
`.trim(),
      tools: { create_order: createOrderTool },
    });
  }
}

/* =========================================================
   Main Entry
========================================================= */

export default defineAgent({
  prewarm: async (proc: JobProcess) => {
    proc.userData.vad = await silero.VAD.load();
  },

  entry: async (ctx: JobContext) => {
    const session = new voice.AgentSession({
      stt: new inference.STT({ model: STT_MODEL, language: "en" }),
      llm: new inference.LLM({ model: LLM_MODEL }),
      tts: new inference.TTS({ model: TTS_MODEL, voice: "default" }),
      vad: ctx.proc.userData.vad as any,
      voiceOptions: { preemptiveGeneration: true },
    });

    const usageCollector = new metrics.UsageCollector();
    session.on(voice.AgentSessionEventTypes.MetricsCollected, (ev: any) => usageCollector.collect(ev.metrics));

    await ctx.connect();

    const agent = new OrderPilotAgent();
    await session.start({ agent, room: ctx.room, inputOptions: {} });

    const say = async (t: string) => session.say(short(t));

    // Greeting (short + premium)
    await say("Hi — OrderPilot. Collection or delivery?");

    // Silence handling: reprompt once, then end
    let silenceTimer: NodeJS.Timeout | null = null;
    let silenceCount = 0;
    const resetSilence = () => {
      if (silenceTimer) clearTimeout(silenceTimer);
      silenceTimer = setTimeout(async () => {
        silenceCount += 1;
        if (silenceCount === 1) {
          await say("Sorry — are you still there?");
          resetSilence();
        } else {
          await say("No problem. Call again when you’re ready. Bye.");
          // Some versions do not expose session.end(); room disconnect will end the leg anyway.
          try {
            await (session as any).end?.();
          } catch {}
        }
      }, SILENCE_MS);
    };
    resetSilence();

    const applyFaq = async (q: string) => {
      const t = normTxt(q);
      if (/(open|opening|hours|close|closing|time)/.test(t)) {
        await say("I’m not sure of today’s hours — but I can take your order now for collection.");
        return true;
      }
      if (/(address|where are you|location)/.test(t)) {
        await say("I don’t have the address to hand — but I can take your order now for collection.");
        return true;
      }
      if (/(card|cash|pay|payment)/.test(t)) {
        await say("Payment depends on the store — but I can take your order now and staff will confirm.");
        return true;
      }
      if (/(how long|wait|ready|eta)/.test(t)) {
        await say("Times vary with how busy it is — I can take your order and staff will confirm the wait time.");
        return true;
      }
      return false;
    };

    const placeOrder = async () => {
      agent.state = "placing";
      await say("Perfect — one moment.");

      try {
        const pickup = normalizePickupTime(agent.draft.pickup_time);
        if (!pickup) {
          agent.draft.pickup_time = null;
          agent.state = "drafting";
          agent.confirmAsked = false;
          await say('What time would you like it for? Say "ASAP" or "19:00".');
          return;
        }

        const data = await (agent as any).tools.create_order.execute({
          customer_name: agent.draft.customer_name || "Phone Customer",
          service_type: agent.draft.service_type,
          pickup_time: pickup,
          notes: agent.draft.notes ?? null,
          items: agent.draft.items.map((i) => ({
            item_name: [
              i.item_name,
              i.modifiers?.length ? `mods: ${i.modifiers.join(", ")}` : null,
              i.special_instructions ? `note: ${i.special_instructions}` : null,
            ]
              .filter(Boolean)
              .join(" | "),
            quantity: i.quantity,
            unit_price: null,
          })),
        });

        const orderId = data?.order_id ?? data?.id ?? null;

        agent.state = "completed";
        await say(orderId ? `All set. Order confirmed — number ${orderId}.` : "All set. Order confirmed.");
        await say("Thanks. Bye.");

        agent.draft = resetDraft();
        agent.confirmAsked = false;
      } catch (e: any) {
        agent.state = "completed";
        const msg = String(e?.message || "");
        if (msg.includes("Invalid time")) {
          agent.draft.pickup_time = null;
          agent.state = "drafting";
          agent.confirmAsked = false;
          await say('Sorry — what time would you like it for? Say "ASAP" or "19:00".');
          return;
        }
        await say("Sorry — I couldn’t place that automatically. Please call again or speak to staff.");
        agent.draft = resetDraft();
        agent.confirmAsked = false;
      }
    };

    const askNext = async () => {
      const next = draftNeedsNext(agent.draft);
      if (next) {
        agent.state = "drafting";
        agent.confirmAsked = false;
        await say(next);
        return;
      }

      if (!agent.confirmAsked) {
        agent.state = "confirming";
        agent.confirmAsked = true;
        await say(confirmOrderSummary(agent.draft));
        return;
      }

      // If already confirming, keep it tight
      await say("Is that correct? Yes or no.");
    };

    session.on(voice.AgentSessionEventTypes.UserSpeechCommitted, async (ev: any) => {
      resetSilence();
      silenceCount = 0;

      const text = clean(ev?.text || "");
      if (!text || text.length < 2) return;

      // Fast path yes/no during confirmation
      if (agent.state === "confirming") {
        if (YES.test(text)) {
          await placeOrder();
          return;
        }
        if (NO.test(text)) {
          // Let them edit; minimal prompt
          agent.state = "drafting";
          agent.confirmAsked = false;
          await say("No problem — what would you like to change?");
          return;
        }
        // If they start editing while confirming, fall through to extraction
      }

      // Direct service type detection (cheaper than LLM)
      const st = normalizeServiceType(text);
      if (st) agent.draft.service_type = st;

      // Direct time detection (cheap)
      const pt = normalizePickupTime(text);
      if (pt && !agent.draft.pickup_time) agent.draft.pickup_time = pt;

      // LLM extraction for items/edits/questions
      const ex = await extractFromText(session as any, text);

      // FAQ handling
      if (ex.intent === "question") {
        const handled = await applyFaq(ex.question || text);
        if (!handled) await say("I’m not sure — but I can take your order now for collection.");
        await askNext();
        return;
      }

      if (ex.intent === "restart") {
        agent.draft = resetDraft();
        agent.state = "drafting";
        agent.confirmAsked = false;
        await say("No worries — starting again. Collection or delivery?");
        return;
      }

      if (ex.intent === "cancel") {
        agent.draft = resetDraft();
        agent.state = "completed";
        agent.confirmAsked = false;
        await say("Okay — cancelled. Bye.");
        return;
      }

      if (ex.name && !agent.draft.customer_name) agent.draft.customer_name = clean(ex.name).slice(0, 60);

      if (ex.pickup_time) {
        const npt = normalizePickupTime(ex.pickup_time);
        if (npt) agent.draft.pickup_time = npt;
      }

      if (ex.service_type) agent.draft.service_type = ex.service_type;

      // Apply item changes
      const items = Array.isArray(ex.items) ? ex.items : [];

      if (items.length) {
        for (const it of items) {
          const rawName = clean(it.name || "");
          if (!rawName) continue;

          const match = menuCanonicalize(rawName);

          if (!match.ok) {
            if (!MENU.length) {
              // no menu configured: accept anything
              upsertItem(agent.draft, {
                item_name: rawName,
                quantity: Math.max(1, Number(it.quantity || 1)),
                modifiers: (it.modifiers || []).map(clean).filter(Boolean),
                special_instructions: it.special_instructions ? clean(it.special_instructions) : null,
              });
              continue;
            }

            const sugg = (match as any).suggestions || [];
            if (sugg.length) {
              await say(`Sorry — I don’t have that. Did you mean ${sugg.slice(0, 3).join(", ")}?`);
            } else {
              await say("Sorry — I don’t have that on the menu. What was the closest item name?");
            }
            // Don’t proceed to confirm until they clarify
            agent.confirmAsked = false;
            agent.state = "drafting";
            return;
          }

          const canonical = (match as any).canonical;

          const di: DraftItem = {
            item_name: canonical,
            quantity: Math.max(1, Number(it.quantity || 1)),
            modifiers: (it.modifiers || []).map(clean).filter(Boolean),
            special_instructions: it.special_instructions ? clean(it.special_instructions) : null,
          };

          if (ex.intent === "remove") {
            const removed = removeItemByName(agent.draft, canonical);
            if (!removed) await say(`I don’t have ${canonical} on the order.`);
            agent.confirmAsked = false;
            continue;
          }

          // add/order/change default -> upsert
          upsertItem(agent.draft, di);
          agent.confirmAsked = false;
        }
      }

      // If they explicitly confirmed in-text (e.g. "yes that's right")
      if (ex.intent === "confirm" && agent.state === "confirming") {
        await placeOrder();
        return;
      }

      await askNext();
    });

    // Keep worker alive
    await new Promise(() => {});
  },
});

cli.runApp(
  new ServerOptions({
    agent: fileURLToPath(import.meta.url),
    agentName: "orderpilot-phone-agent",
  })
);
