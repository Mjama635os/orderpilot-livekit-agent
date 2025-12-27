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

import * as livekit from "@livekit/agents-plugin-livekit";
import * as silero from "@livekit/agents-plugin-silero";
import * as openai from "@livekit/agents-plugin-openai";

import { z } from "zod";
import { fetch } from "undici";
import { fileURLToPath } from "node:url";

/* =========================================================
   OrderPilot – Production Menu-Aware Phone Agent (Build-Safe)
   FIXES:
   - NEVER send ISO pickup_time to backend (only "ASAP" or "HH:MM")
   - Prevent LLM from tool-calling create_order (we call tool execute directly)
   - Backend timeout + 1 retry on timeout/network/5xx
   ========================================================= */

/* =========================
   Types
========================= */

type ServiceType = "collection" | "delivery";

type MenuCategory =
  | "pizza"
  | "starter"
  | "salad"
  | "calzone"
  | "pasta"
  | "dessert"
  | "icecream"
  | "drink"
  | "dip"
  | "deal"
  | "offer"
  | "custom";

type PizzaSize = 10 | 13 | 15;

type MenuItem = {
  category: MenuCategory;
  canonical: string;
  display: string;
  synonyms?: string[];
  requires?: { size?: boolean };
};

type DraftItem = {
  category: MenuCategory;
  canonical: string;
  display: string;
  quantity: number;
  size?: PizzaSize;
  modifiers: string[];
  notes?: string | null;
  unit_price?: number | null;
};

type PendingQuestion = "item" | "name" | "time" | "pizza_size" | null;

type OrderDraft = {
  service_type: ServiceType; // locked to collection for week 1
  customer_name: string | null;
  pickup_time: string | null; // MUST be "ASAP" | "HH:MM" for backend
  items: DraftItem[];
  notes: string | null;

  pending_question: PendingQuestion;
  reprompt_count: number;

  pending_unknown_item: string | null;
  pending_unknown_attempts: number;
};

type AgentState = "drafting" | "confirming" | "placing" | "completed";

/* =========================
   Constants
========================= */

const YES =
  /\b(yes|yeah|yep|yup|ok|okay|correct|confirm|sounds good|go ahead|that's right|thats right|perfect)\b/i;
const NO = /\b(no|nope|wrong|change|cancel|restart|start again|not that)\b/i;

const MAX_TURN_CHARS = 240;

/* =========================
   Helpers
========================= */

function short(text: string): string {
  const t = text.trim().replace(/\s+/g, " ");
  return t.length > MAX_TURN_CHARS ? `${t.slice(0, MAX_TURN_CHARS - 1)}…` : t;
}

function clean(text: string): string {
  return text.trim().replace(/\s+/g, " ");
}

function resetDraft(): OrderDraft {
  return {
    service_type: "collection",
    customer_name: null,
    pickup_time: null,
    items: [],
    notes: null,
    pending_question: null,
    reprompt_count: 0,
    pending_unknown_item: null,
    pending_unknown_attempts: 0,
  };
}

function normalizeTime(input: string): string | null {
  const t = input.toLowerCase().trim();
  if (!t) return null;

  if (/(asap|now|as soon)/.test(t)) return "ASAP";
  const inMin = t.match(/in\s+(\d+)\s*(min|mins|minutes?)/);
  if (inMin) return "ASAP";

  // “half seven” -> assume evening for takeaway
  const half = t.match(/half\s+([a-z]+|\d{1,2})/);
  if (half) {
    const raw = half[1];
    const wordMap: Record<string, number> = {
      one: 1,
      two: 2,
      three: 3,
      four: 4,
      five: 5,
      six: 6,
      seven: 7,
      eight: 8,
      nine: 9,
      ten: 10,
      eleven: 11,
      twelve: 12,
    };
    const base = /^\d{1,2}$/.test(raw) ? Number(raw) : wordMap[raw] ?? NaN;
    if (!Number.isNaN(base)) {
      const h = base >= 12 ? base : base + 12;
      return `${String(h).padStart(2, "0")}:30`;
    }
  }

  // 7, 7:30, 7pm, 19:00
  const exact = t.match(/\b(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\b/);
  if (exact) {
    let h = Number(exact[1]);
    const m = exact[2] ? Number(exact[2]) : 0;
    const ap = exact[3];

    if (!ap) {
      if (h >= 1 && h <= 11) h += 12;
    } else {
      if (ap === "pm" && h < 12) h += 12;
      if (ap === "am" && h === 12) h = 0;
    }

    return `${String(h).padStart(2, "0")}:${String(m).padStart(2, "0")}`;
  }

  return null;
}

/**
 * Backend guardrail: ONLY "ASAP" or "HH:MM" allowed.
 * If anything else (including ISO timestamps) -> fallback to ASAP.
 */
function ensureBackendPickupTime(value: string | null): string {
  if (!value) return "ASAP";
  const v = value.trim();

  if (/^ASAP$/i.test(v)) return "ASAP";

  const m = v.match(/^(\d{1,2}):(\d{2})$/);
  if (m) {
    const hh = Math.min(23, Math.max(0, Number(m[1])));
    const mm = Math.min(59, Math.max(0, Number(m[2])));
    return `${String(hh).padStart(2, "0")}:${String(mm).padStart(2, "0")}`;
  }

  return "ASAP";
}

function parsePizzaSize(text: string): PizzaSize | null {
  const t = text.toLowerCase();
  if (/(10\s*("|inch)\b|ten\s*inch)/.test(t)) return 10;
  if (/(13\s*("|inch)\b|thirteen\s*inch)/.test(t)) return 13;
  if (/(15\s*("|inch)\b|fifteen\s*inch)/.test(t)) return 15;
  return null;
}

function totalItemCount(items: DraftItem[]): number {
  return items.reduce((sum, i) => sum + (i.quantity || 0), 0);
}

function formatItem(i: DraftItem): string {
  const size = i.size ? ` ${i.size}"` : "";
  const mods = i.modifiers?.length ? ` (${i.modifiers.join(", ")})` : "";
  const note = i.notes?.trim() ? ` [${i.notes.trim()}]` : "";
  return `${i.quantity} ${i.display}${size}${mods}${note}`;
}

function fingerprintDraft(d: OrderDraft): string {
  const items = [...d.items]
    .sort((a, b) => (a.canonical + (a.size ?? "")).localeCompare(b.canonical + (b.size ?? "")))
    .map((i) => ({
      c: i.canonical,
      q: i.quantity,
      s: i.size ?? null,
      m: [...(i.modifiers || [])].sort(),
      n: (i.notes || "").trim(),
    }));
  return JSON.stringify({
    name: (d.customer_name || "").trim(),
    time: d.pickup_time || "",
    items,
    notes: (d.notes || "").trim(),
  });
}

function confirmSummary(d: OrderDraft): string {
  const items = d.items.map(formatItem).join(", ");
  const count = totalItemCount(d.items);
  const time = ensureBackendPickupTime(d.pickup_time);
  const name = d.customer_name ?? "your name";
  return short(
    `Just to confirm: ${items}. Total ${count} item${count === 1 ? "" : "s"}. Collection at ${time}, under ${name}. Is that correct?`
  );
}

/* =========================
   FAQ Handling
========================= */

const FAQ: Record<string, string> = {
  hours: "I’m not 100% sure of today’s hours. If you tell me your order, I can place it now.",
  address: "I don’t have the full address here. If you place the order, staff can confirm details if needed.",
  payment: "Most customers pay by card or cash at collection. If you prefer one, tell me and I’ll note it.",
  delivery:
    "At the moment I can take collection orders. If you need delivery, I can still take the order and staff will confirm.",
  wait: "Wait time depends on how busy it is. If you tell me what you want, I’ll place it now.",
};

function classifyFaq(q: string): keyof typeof FAQ | null {
  const t = q.toLowerCase();
  if (/(open|close|closing|hours|what time are you open)/.test(t)) return "hours";
  if (/(where are you|address|located|postcode)/.test(t)) return "address";
  if (/(card|cash|pay|payment|apple pay|google pay)/.test(t)) return "payment";
  if (/(deliver|delivery)/.test(t)) return "delivery";
  if (/(how long|wait|ready|eta|time will it take)/.test(t)) return "wait";
  return null;
}

/* =========================
   Menu
========================= */

function nkey(s: string): string {
  return s
    .toLowerCase()
    .replace(/[^a-z0-9\s"]/g, "")
    .replace(/\s+/g, " ")
    .trim();
}

const MENU: MenuItem[] = [
  // Pizzas (size required)
  ...[
    "Margherita",
    "Hawaiian",
    "Napoletana",
    "Calabrra",
    "Vegetarian",
    "Vegetarian Hot",
    "Roasted Vegetarian",
    "Pepperoni Plus",
    "Quatro Stagioni",
    "American Hot",
    "Mexican Hot",
    "Siciliana",
    "Chicken Tikka",
    "Quatro Formaggi",
    "Orlando Pizza",
    "Bbq Sweet",
    "Meaty Pizza",
    "Diavola",
    "Rughetta Pizza",
    "Piccante",
    "Seafood",
    "La Venice Special",
    "California",
    "Hot Tonno",
    "Indiana",
    "Chicken Supreme",
    "Garlic Meat Lover",
    "Parmiggiana",
    "Free Choice",
    "Half And Half",
  ].map((name) => ({
    category: "pizza" as const,
    canonical: nkey(name),
    display: name,
    requires: { size: true },
    synonyms: [nkey(name)],
  })),

  // Drinks
  ...["Coke", "Diet Coke", "Fanta", "Sprite", "Water"].map((name) => ({
    category: "drink" as const,
    canonical: nkey(name),
    display: name,
    synonyms: [nkey(name)],
  })),

  // Dips
  ...["Garlic Dip", "Chilli Dip", "BBQ Dip", "Mayo"].map((name) => ({
    category: "dip" as const,
    canonical: nkey(name),
    display: name,
    synonyms: [nkey(name)],
  })),
];

const MENU_BY_CANONICAL = new Map<string, MenuItem>(MENU.map((m) => [m.canonical, m]));
const ALL_SEARCH_TERMS: { canonical: string; key: string }[] = MENU.flatMap((m) => {
  const keys = new Set<string>([m.display, m.canonical, ...(m.synonyms || [])].map((x) => nkey(x)));
  return [...keys].filter(Boolean).map((k) => ({ canonical: m.canonical, key: k }));
});

function tokenSet(s: string): Set<string> {
  return new Set(nkey(s).split(" ").filter(Boolean));
}

function scoreMatch(query: string, candidate: string): number {
  const q = tokenSet(query);
  const c = tokenSet(candidate);
  if (!q.size || !c.size) return 0;
  let inter = 0;
  for (const w of q) if (c.has(w)) inter++;
  return inter / Math.max(q.size, c.size);
}

function bestMenuMatches(query: string, topK = 3): { item: MenuItem; score: number }[] {
  const q = nkey(query);
  const scored = ALL_SEARCH_TERMS.map((t) => ({
    canonical: t.canonical,
    score: scoreMatch(q, t.key),
  }))
    .sort((a, b) => b.score - a.score)
    .slice(0, 12);

  const uniq: { canonical: string; score: number }[] = [];
  for (const s of scored) {
    if (!uniq.find((u) => u.canonical === s.canonical)) uniq.push(s);
    if (uniq.length >= topK) break;
  }

  return uniq
    .map((u) => ({ item: MENU_BY_CANONICAL.get(u.canonical)!, score: u.score }))
    .filter((x) => !!x.item);
}

function upsertDraftItem(d: OrderDraft, item: DraftItem): void {
  const key = item.canonical + (item.size ? `|${item.size}` : "");
  const idx = d.items.findIndex((x) => (x.canonical + (x.size ? `|${x.size}` : "")) === key);

  if (idx === -1) {
    d.items.push(item);
    return;
  }

  d.items[idx].quantity += item.quantity;
  d.items[idx].modifiers = Array.from(
    new Set([...(d.items[idx].modifiers || []), ...(item.modifiers || [])])
  );
  const a = d.items[idx].notes?.trim() || "";
  const b = item.notes?.trim() || "";
  d.items[idx].notes = a && b ? `${a}; ${b}` : a || b || null;
}

function removeByText(d: OrderDraft, text: string): boolean {
  const matches = bestMenuMatches(text, 1);
  if (!matches.length) return false;
  const m = matches[0].item;
  const before = d.items.length;
  d.items = d.items.filter((x) => x.canonical !== m.canonical);
  return d.items.length !== before;
}

function changeQtyByText(d: OrderDraft, text: string, qty: number): boolean {
  const matches = bestMenuMatches(text, 1);
  if (!matches.length) return false;
  const m = matches[0].item;
  const idx = d.items.findIndex((x) => x.canonical === m.canonical);
  if (idx === -1) return false;
  d.items[idx].quantity = qty;
  return true;
}

function clearPending(d: OrderDraft) {
  d.pending_question = null;
  d.reprompt_count = 0;
}

function getNextQuestionAndSetPending(d: OrderDraft): string | null {
  if (d.pending_question) return null;

  if (!d.items.length) {
    d.pending_question = "item";
    return "What would you like to order?";
  }

  if (!d.customer_name) {
    d.pending_question = "name";
    return "What name is the order under?";
  }

  if (!d.pickup_time) {
    d.pending_question = "time";
    return "What time would you like it for? ASAP or a time like 19:30.";
  }

  const pizzaMissingSize = d.items.find((i) => i.category === "pizza" && !i.size);
  if (pizzaMissingSize) {
    d.pending_question = "pizza_size";
    return `For the ${pizzaMissingSize.display}—10, 13, or 15 inch?`;
  }

  return null;
}

/* =========================
   LLM parsing (optional) – no tools
========================= */

const ParseSchema = z.object({
  intent: z.enum(["order", "add", "remove", "change", "confirm", "cancel", "question", "unknown"]),
  name: z.string().nullable().optional(),
  pickup_time: z.string().nullable().optional(),
  raw_items: z
    .array(
      z.object({
        text: z.string(),
        quantity: z.number().int().min(1).default(1),
        size: z.number().int().optional().nullable(),
        modifiers: z.array(z.string()).default([]),
        notes: z.string().nullable().optional(),
      })
    )
    .optional(),
  remove_texts: z.array(z.string()).optional(),
  change_qty: z
    .object({
      text: z.string(),
      quantity: z.number().int().min(1),
    })
    .nullable()
    .optional(),
  notes: z.string().nullable().optional(),
});

type Parsed = z.infer<typeof ParseSchema>;

function shouldUseLLM(text: string, draft: OrderDraft): boolean {
  const t = text.toLowerCase();
  if (YES.test(t) || NO.test(t)) return false;
  if (classifyFaq(t)) return false;
  if (draft.pending_question) return false;
  if (draft.pending_unknown_item) return false;

  if (/(,| and | plus | also )/.test(t)) return true;
  if (/\b(no|without|extra|add|remove|change|instead)\b/.test(t)) return true;
  if (/\b(\d+|one|two|three|four|five)\b/.test(t) && t.split(/\s+/).length >= 5) return true;

  return t.trim().split(/\s+/).length >= 10;
}

async function parseWithLLM(session: voice.AgentSession<any>, text: string): Promise<Parsed | null> {
  try {
    const prompt = `
Return ONLY valid JSON for this UK takeaway utterance.

Keys:
intent: order/add/remove/change/confirm/cancel/question/unknown
name: string|null
pickup_time: string|null (as spoken)
raw_items: [{text, quantity, size(10/13/15|null), modifiers[], notes|null}]
remove_texts: string[]
change_qty: {text, quantity} | null
notes: string|null

Utterance:
${text}
`.trim();

    const reply = await session.chat({ messages: [{ role: "user", content: prompt }] });
    const raw = typeof reply?.content === "string" ? reply.content : JSON.stringify(reply?.content ?? "");
    const jsonText = raw.trim().replace(/^```json\s*/i, "").replace(/```$/i, "").trim();
    const parsed = JSON.parse(jsonText);
    return ParseSchema.parse(parsed);
  } catch {
    return null;
  }
}

/* =========================
   Backend tool (manual execute only)
========================= */

const createOrderTool = llm.tool({
  description: "Create a confirmed restaurant order in the OrderPilot backend. Manual execute only.",
  parameters: z.object({
    customer_name: z.string(),
    service_type: z.enum(["collection", "delivery"]),
    pickup_time: z.string(),
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
    const url = process.env.ORDERPILOT_ORDERS_URL;
    const restaurantId = process.env.DEFAULT_RESTAURANT_ID;
    if (!url) throw new Error("Missing ORDERPILOT_ORDERS_URL");
    if (!restaurantId) throw new Error("Missing DEFAULT_RESTAURANT_ID");

    const payload = {
      restaurant_id: restaurantId,
      customer_name: args.customer_name,
      customer_phone: null,
      pickup_time: args.pickup_time,
      service_type: args.service_type,
      notes: args.notes ?? null,
      items: args.items,
    };

    const attempt = async (): Promise<any> => {
      const controller = new AbortController();
      const timeout = setTimeout(() => controller.abort(), 8000);

      try {
        const res = await fetch(url, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
          signal: controller.signal,
        });

        const data = await res.json().catch(() => ({}));
        if (!res.ok) {
          const err = new Error(
            `Order create failed ${res.status}: ${JSON.stringify(data).slice(0, 200)}`
          );
          (err as any).status = res.status;
          throw err;
        }

        return data;
      } finally {
        clearTimeout(timeout);
      }
    };

    try {
      return await attempt();
    } catch (e: any) {
      const status = Number(e?.status || 0);
      const retryable =
        e?.name === "AbortError" || status === 0 || (status >= 500 && status <= 599);
      if (!retryable) throw e;
      return await attempt(); // 1 retry only
    }
  },
});

/* =========================
   Agent
========================= */

class OrderPilotAgent extends voice.Agent {
  draft: OrderDraft;
  state: AgentState;
  confirmAsked: boolean;
  lastConfirmFingerprint: string | null;

  constructor() {
    // IMPORTANT: do NOT register create_order as an LLM tool
    // This prevents the LLM from calling it and inventing ISO times.
    super({
      instructions: `
You are the restaurant phone assistant. Sound like calm, efficient staff.

STYLE:
- Friendly, quick, confident.
- Short answers. One question at a time.

RULES:
- COLLECTION ONLY.
- Confirm once with an itemised summary.
- Unknown items: clarify once; if still unclear, accept as "custom item" and continue.
- If asked a question you don't know, answer politely and move back to ordering.
      `.trim(),
      tools: {}, // <-- prevents LLM tool calls
    });

    this.draft = resetDraft();
    this.state = "drafting";
    this.confirmAsked = false;
    this.lastConfirmFingerprint = null;
  }
}

/* =========================
   Main
========================= */

export default defineAgent({
  prewarm: async (proc: JobProcess) => {
    proc.userData.vad = await silero.VAD.load();
  },

  entry: async (ctx: JobContext) => {
    const session = new voice.AgentSession({
      stt: new openai.STT({ model: "whisper-1", language: "en" }),
      llm: new inference.LLM({ model: "openai/gpt-4o-mini" }),
      tts: new openai.TTS({ model: "tts-1", voice: "nova" }),
      turnDetection: new livekit.turnDetector.MultilingualModel(),
      vad: ctx.proc.userData.vad as silero.VAD,
      voiceOptions: { preemptiveGeneration: true },
    });

    const usageCollector = new metrics.UsageCollector();
    session.on(voice.AgentSessionEventTypes.MetricsCollected, (ev) => usageCollector.collect(ev.metrics));

    await ctx.connect();

    const agent = new OrderPilotAgent();
    await session.start({ agent, room: ctx.room, inputOptions: {} });

    const say = async (t: string) => session.say(short(t));

    // Silence guard
    let silenceTimer: NodeJS.Timeout | null = null;
    let silenceCount = 0;
    const resetSilence = () => {
      if (silenceTimer) clearTimeout(silenceTimer);
      silenceTimer = setTimeout(async () => {
        silenceCount += 1;
        if (silenceCount === 1) {
          await say("Sorry—are you still there?");
          resetSilence();
        } else {
          await say("No problem—call again when you’re ready. Bye.");
        }
      }, 12000);
    };
    resetSilence();

    const markDraftChanged = () => {
      agent.confirmAsked = false;
      agent.lastConfirmFingerprint = null;
      if (agent.state === "confirming") agent.state = "drafting";
    };

    const placeOrder = async () => {
      agent.state = "placing";
      await say("Perfect—one second.");

      try {
        const data = await createOrderTool.execute({
          customer_name: agent.draft.customer_name!,
          service_type: "collection",
          pickup_time: ensureBackendPickupTime(agent.draft.pickup_time),
          notes: agent.draft.notes ?? null,
          items: agent.draft.items.map((i) => ({
            item_name: [
              i.display,
              i.size ? `${i.size}"` : null,
              i.modifiers?.length ? `mods: ${i.modifiers.join(", ")}` : null,
              i.notes?.trim() ? `note: ${i.notes.trim()}` : null,
            ]
              .filter(Boolean)
              .join(" | "),
            quantity: i.quantity,
            unit_price: i.unit_price ?? null,
          })),
        });

        const orderId = data?.order_id ?? data?.id ?? null;
        agent.state = "completed";
        await say(orderId ? `All done—order confirmed. Order number ${orderId}.` : "All done—order confirmed.");
        await say("Thanks. Bye.");
      } catch (e: any) {
        agent.state = "completed";
        await say("Sorry—I couldn’t place that automatically just now. Please call again or speak to staff.");
      } finally {
        agent.draft = resetDraft();
        agent.confirmAsked = false;
        agent.lastConfirmFingerprint = null;
      }
    };

    await say("Hi—OrderPilot. What can I get you today?");

    session.on(voice.AgentSessionEventTypes.SpeechCommitted, async (ev: any) => {
      resetSilence();
      silenceCount = 0;

      // Ignore agent speech
      if (ev?.participant?.identity && String(ev.participant.identity).startsWith("agent-")) return;

      const text = clean(String(ev?.text || ""));
      if (!text || text.length < 2) return;

      // FAQ
      const faq = classifyFaq(text);
      if (faq) {
        await say(`${FAQ[faq]} What would you like to order?`);
        return;
      }

      // Delivery safe mode
      if (/\b(delivery|deliver)\b/i.test(text)) {
        agent.draft.service_type = "collection";
        markDraftChanged();
        await say("No problem. I can take collection now—what would you like?");
        return;
      }

      // Confirmation yes/no
      if (YES.test(text)) {
        if (agent.state === "confirming") return placeOrder();
      }
      if (NO.test(text)) {
        if (/restart|start again/i.test(text)) {
          agent.draft = resetDraft();
          agent.state = "drafting";
          agent.confirmAsked = false;
          agent.lastConfirmFingerprint = null;
          await say("No problem—start again. What would you like?");
          return;
        }
        if (agent.state === "confirming") {
          agent.state = "drafting";
          agent.confirmAsked = false;
          agent.lastConfirmFingerprint = null;
          await say("No worries—what would you like to change?");
          return;
        }
      }

      // Slot-lock fast path
      if (agent.draft.pending_question === "name") {
        const maybeName = text.split(/\s+/).slice(0, 3).join(" ");
        if (maybeName.length >= 2) {
          agent.draft.customer_name = maybeName;
          clearPending(agent.draft);
          markDraftChanged();
        } else {
          agent.draft.reprompt_count++;
          await say("Sorry—what name is the order under?");
          return;
        }
      }

      if (agent.draft.pending_question === "time") {
        const nt = normalizeTime(text);
        if (nt) {
          agent.draft.pickup_time = nt; // normalized to ASAP/HH:MM
          clearPending(agent.draft);
          markDraftChanged();
        } else {
          agent.draft.reprompt_count++;
          await say("Sorry—ASAP, or a time like 19:30?");
          return;
        }
      }

      if (agent.draft.pending_question === "pizza_size") {
        const s = parsePizzaSize(text);
        if (s) {
          const p = agent.draft.items.find((i) => i.category === "pizza" && !i.size);
          if (p) p.size = s;
          clearPending(agent.draft);
          markDraftChanged();
        } else {
          agent.draft.reprompt_count++;
          await say("Sorry—10, 13, or 15 inch?");
          return;
        }
      }

      // Heuristic time capture
      if (!agent.draft.pickup_time) {
        const nt = normalizeTime(text);
        if (nt) {
          agent.draft.pickup_time = nt; // normalized
          markDraftChanged();
        }
      }

      // Heuristic name capture
      if (!agent.draft.customer_name) {
        const m = text.match(/\b(it'?s|under|name is)\s+([a-zA-Z]{2,})\b/i);
        if (m?.[2]) {
          agent.draft.customer_name = m[2];
          markDraftChanged();
        }
      }

      // Remove / qty edits
      const removeHit = text.match(/\b(remove|take off|delete)\s+(.*)$/i);
      if (removeHit?.[2]) {
        const ok = removeByText(agent.draft, removeHit[2]);
        if (ok) {
          markDraftChanged();
          await say("Done. Anything else?");
        } else {
          await say("Sorry—what should I remove?");
        }
        return;
      }

      const qtyHit = text.match(/\b(make it|change to|make that)\s+(\d+)\s+(.*)$/i);
      if (qtyHit?.[2] && qtyHit?.[3]) {
        const qty = Number(qtyHit[2]);
        if (Number.isFinite(qty) && qty >= 1) {
          const ok = changeQtyByText(agent.draft, qtyHit[3], qty);
          if (ok) {
            markDraftChanged();
            await say("Done. Anything else?");
            return;
          }
        }
      }

      // Unknown item pending
      if (agent.draft.pending_unknown_item) {
        const matches = bestMenuMatches(text, 3);
        const top = matches[0];
        const score = top?.score ?? 0;

        if (matches.length && score >= 0.32) {
          const chosen = top.item;
          const size = parsePizzaSize(text);

          upsertDraftItem(agent.draft, {
            category: chosen.category,
            canonical: chosen.canonical,
            display: chosen.display,
            quantity: 1,
            size: chosen.category === "pizza" ? (size ?? undefined) : undefined,
            modifiers: [],
            notes: null,
            unit_price: null,
          });

          agent.draft.pending_unknown_item = null;
          agent.draft.pending_unknown_attempts = 0;
          clearPending(agent.draft);
          markDraftChanged();
        } else {
          agent.draft.pending_unknown_attempts += 1;
          if (agent.draft.pending_unknown_attempts >= 2) {
            upsertDraftItem(agent.draft, {
              category: "custom",
              canonical: nkey(agent.draft.pending_unknown_item),
              display: agent.draft.pending_unknown_item,
              quantity: 1,
              modifiers: [],
              notes: "CUSTOM ITEM (not in menu)",
              unit_price: null,
            });

            agent.draft.pending_unknown_item = null;
            agent.draft.pending_unknown_attempts = 0;
            clearPending(agent.draft);
            markDraftChanged();
            await say("Okay—got it.");
          } else {
            await say("Sorry—what was that item? You can say a pizza name, or a drink like Coke.");
          }
          return;
        }
      }

      // Optional LLM parse (never tool calls)
      let parsed: Parsed | null = null;
      if (shouldUseLLM(text, agent.draft)) {
        parsed = await parseWithLLM(session, text);
      }

      if (parsed?.intent === "question") {
        const fq2 = classifyFaq(text);
        if (fq2) await say(`${FAQ[fq2]} What would you like to order?`);
        else await say("I’m not fully sure. I can take your order now—what would you like?");
        return;
      }

      if (parsed?.intent === "cancel") {
        agent.draft = resetDraft();
        agent.state = "drafting";
        agent.confirmAsked = false;
        agent.lastConfirmFingerprint = null;
        await say("Okay—cancelled. What would you like to order?");
        return;
      }

      if (parsed?.notes?.trim()) {
        const n = parsed.notes.trim();
        agent.draft.notes = agent.draft.notes ? `${agent.draft.notes}; ${n}` : n;
        markDraftChanged();
      }

      if (parsed?.name?.trim()) {
        agent.draft.customer_name = parsed.name.trim();
        clearPending(agent.draft);
        markDraftChanged();
      }

      if (parsed?.pickup_time && !agent.draft.pickup_time) {
        const nt = normalizeTime(parsed.pickup_time);
        if (nt) {
          agent.draft.pickup_time = nt; // normalized
          clearPending(agent.draft);
          markDraftChanged();
        }
      }

      if (parsed?.remove_texts?.length) {
        let removed = false;
        for (const r of parsed.remove_texts) removed = removeByText(agent.draft, r) || removed;
        if (removed) markDraftChanged();
      }

      if (parsed?.change_qty?.text && parsed.change_qty.quantity) {
        const ok = changeQtyByText(agent.draft, parsed.change_qty.text, parsed.change_qty.quantity);
        if (ok) markDraftChanged();
      }

      // Add items from parsed or treat as phrase
      const rawItems =
        parsed?.raw_items?.length
          ? parsed.raw_items
          : [
              {
                text,
                quantity: 1,
                size: parsePizzaSize(text) ?? null,
                modifiers: [],
                notes: null,
              },
            ];

      let addedSomething = false;

      for (const r of rawItems) {
        const phrase = clean(r.text);
        if (!phrase) continue;

        if (/^my name is\b/i.test(phrase)) continue;
        if (normalizeTime(phrase)) continue;

        const matches = bestMenuMatches(phrase, 3);
        const top = matches[0];
        const score = top?.score ?? 0;

        if (!matches.length || score < 0.32) {
          agent.draft.pending_unknown_item = phrase;
          agent.draft.pending_unknown_attempts = 0;
          agent.draft.pending_question = "item";
          agent.draft.reprompt_count = 0;
          await say("Sorry—what was that item? You can say a pizza name, or a drink like Coke.");
          return;
        }

        if (matches.length > 1 && score < 0.55) {
          await say(`Did you mean ${matches.map((m) => m.item.display).join(", ")}?`);
        }

        const chosen = top.item;
        const size = (r.size as PizzaSize | null) ?? parsePizzaSize(phrase) ?? null;

        upsertDraftItem(agent.draft, {
          category: chosen.category,
          canonical: chosen.canonical,
          display: chosen.display,
          quantity: r.quantity || 1,
          size: chosen.category === "pizza" ? (size ?? undefined) : undefined,
          modifiers: r.modifiers || [],
          notes: r.notes ?? null,
          unit_price: null,
        });

        addedSomething = true;
        markDraftChanged();
      }

      if (agent.draft.pending_question === "item" && addedSomething) {
        clearPending(agent.draft);
      }

      // Ask next missing field
      const next = getNextQuestionAndSetPending(agent.draft);
      if (next) {
        agent.state = "drafting";
        await say(next);
        return;
      }

      // Confirm
      const fp = fingerprintDraft(agent.draft);

      if (!agent.confirmAsked || agent.lastConfirmFingerprint !== fp) {
        agent.state = "confirming";
        agent.confirmAsked = true;
        agent.lastConfirmFingerprint = fp;
        await say(confirmSummary(agent.draft));
        return;
      }

      // Talking during confirming -> edit assist
      if (agent.state === "confirming" && !YES.test(text) && !NO.test(text)) {
        agent.state = "drafting";
        agent.confirmAsked = false;
        agent.lastConfirmFingerprint = null;
        await say("No worries—tell me what to change. For example: add Coke, remove garlic dip, or change the time.");
        return;
      }

      await say("Got it.");
    });

    await new Promise(() => {});
  },
});

cli.runApp(new ServerOptions({ agent: fileURLToPath(import.meta.url) }));
