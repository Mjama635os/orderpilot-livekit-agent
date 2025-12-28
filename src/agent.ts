/**
 * OrderPilot LiveKit Voice Agent (Node.js)
 * Production-focused, low-latency restaurant ordering agent.
 *
 * - Uses LiveKit Agents (@livekit/agents) + LiveKit plugin turn detector + Silero VAD
 * - Uses LiveKit Inference model strings for STT/LLM/TTS (Deepgram/ElevenLabs/Cartesia/etc.)
 * - Uses LLM tool-calls to manage a small state machine (drafting → confirming → placing → completed)
 * - Posts orders to ORDERPILOT_ORDERS_URL with strict pickup_time formatting: "ASAP" or "HH:MM"
 */

import dotenv from "dotenv";
dotenv.config({ path: ".env.local" });

import { fetch } from "undici";
import { z } from "zod";

import {
  ServerOptions,
  cli,
  defineAgent,
  llm,
  voice,
} from "@livekit/agents";

import * as livekit from "@livekit/agents-plugin-livekit";
import * as silero from "@livekit/agents-plugin-silero";

import { fileURLToPath } from "node:url";

/* =========================
   Env / Config
========================= */

const ORDERPILOT_ORDERS_URL = mustGetEnv("ORDERPILOT_ORDERS_URL"); // e.g. https://api.orderpilot.co.uk/orders
const DEFAULT_RESTAURANT_ID = mustGetEnv("DEFAULT_RESTAURANT_ID");

const RESTAURANT_NAME = process.env.RESTAURANT_NAME?.trim() || "the restaurant";
const RESTAURANT_ADDRESS = process.env.RESTAURANT_ADDRESS?.trim() || ""; // optional for FAQs
const RESTAURANT_PHONE = process.env.RESTAURANT_PHONE?.trim() || ""; // optional for handoff
const RESTAURANT_OPENING_HOURS = process.env.RESTAURANT_OPENING_HOURS?.trim() || ""; // optional for FAQs

// Models (use LiveKit Inference model strings)
const LLM_MODEL = process.env.LLM_MODEL?.trim() || "openai/gpt-4.1-mini";
const STT_MODEL = process.env.STT_MODEL?.trim() || "deepgram/nova-2-phonecall:en";
// If you don't have Cartesia, use ElevenLabs (supported by LiveKit Inference).
const TTS_MODEL = process.env.TTS_MODEL?.trim() || "elevenlabs/eleven_turbo_v2_5:default";

// Menu: Provide either MENU_ITEMS (comma-separated) or MENU_JSON (array of strings / objects)
const MENU_ITEMS = loadMenuItems();

/* =========================
   Types / State
========================= */

type ServiceType = "collection" | "delivery";
type AgentState = "drafting" | "confirming" | "placing" | "completed";

type DraftItem = {
  item_name: string;
  quantity: number;
  modifiers: string[];
  special_instructions?: string | null;
};

type OrderDraft = {
  state: AgentState;
  service_type: ServiceType | null;
  customer_name: string | null;
  pickup_time: string | null; // MUST be "ASAP" or "HH:MM"
  delivery_address: string | null;
  notes: string | null;
  items: DraftItem[];
};

function newDraft(): OrderDraft {
  return {
    state: "drafting",
    service_type: "collection",
    customer_name: null,
    pickup_time: "ASAP",
    delivery_address: null,
    notes: null,
    items: [],
  };
}

function mustGetEnv(key: string): string {
  const v = process.env[key]?.trim();
  if (!v) throw new Error(`Missing required env var: ${key}`);
  return v;
}

/* =========================
   Helpers: Time normalization
========================= */

function normalizePickupTime(input: unknown): string | null {
  if (input === null || input === undefined) return null;
  const raw = String(input).trim();
  if (!raw) return null;

  const s = raw.toLowerCase();

  // ASAP
  if (s === "asap" || s === "now" || s === "soon") return "ASAP";

  // "in 20 minutes" -> "ASAP" (backend supports this phrase too, but keep stable)
  if (/^in\s+\d+\s*(min|mins|minute|minutes)\b/.test(s)) return "ASAP";

  // "half seven" (UK) -> 19:30 by default if "seven" and no am/pm, assume evening for takeaways
  if (s.includes("half") && (s.includes("seven") || s.includes("six") || s.includes("eight") || s.includes("nine"))) {
    const hour = wordHourToNumber(s);
    if (hour !== null) {
      const hh = inferEveningHour(hour, s);
      return `${String(hh).padStart(2, "0")}:30`;
    }
  }

  // HH:MM
  const hhmm = raw.match(/^(\d{1,2}):(\d{2})$/);
  if (hhmm) {
    const hh = clampInt(Number(hhmm[1]), 0, 23);
    const mm = clampInt(Number(hhmm[2]), 0, 59);
    return `${String(hh).padStart(2, "0")}:${String(mm).padStart(2, "0")}`;
  }

  // "7pm" / "7 pm" / "7:30pm"
  const ampm = s.match(/^(\d{1,2})(?::(\d{2}))?\s*(am|pm)$/);
  if (ampm) {
    let hh = clampInt(Number(ampm[1]), 0, 23);
    const mm = clampInt(ampm[2] ? Number(ampm[2]) : 0, 0, 59);
    const ap = ampm[3];
    if (ap === "pm" && hh < 12) hh += 12;
    if (ap === "am" && hh === 12) hh = 0;
    return `${String(hh).padStart(2, "0")}:${String(mm).padStart(2, "0")}`;
  }

  // "7" or "7:30" without am/pm: assume evening for takeaways (17:00–23:00)
  const bare = s.match(/^(\d{1,2})(?::(\d{2}))?$/);
  if (bare) {
    const hour = clampInt(Number(bare[1]), 0, 23);
    const minute = clampInt(bare[2] ? Number(bare[2]) : 0, 0, 59);
    const hh = inferEveningHour(hour, s);
    return `${String(hh).padStart(2, "0")}:${String(minute).padStart(2, "0")}`;
  }

  // ISO timestamp: convert to HH:MM local time (server locale)
  const d = new Date(raw);
  if (!Number.isNaN(d.getTime())) {
    return `${String(d.getHours()).padStart(2, "0")}:${String(d.getMinutes()).padStart(2, "0")}`;
  }

  return null;
}

function clampInt(n: number, min: number, max: number): number {
  if (Number.isNaN(n)) return min;
  return Math.min(max, Math.max(min, Math.trunc(n)));
}

function inferEveningHour(hour: number, sLower: string): number {
  // If user explicitly says morning/afternoon/evening, respect it
  if (sLower.includes("am")) return hour === 12 ? 0 : hour;
  if (sLower.includes("pm")) return hour < 12 ? hour + 12 : hour;

  // Heuristic: assume evening (UK takeaways), so 7 -> 19, 8 -> 20, 9 -> 21, 10 -> 22, 11 -> 23
  if (hour >= 1 && hour <= 11) return hour + 12;
  return hour;
}

function wordHourToNumber(sLower: string): number | null {
  const map: Record<string, number> = {
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
  for (const k of Object.keys(map)) {
    if (sLower.includes(k)) return map[k];
  }
  return null;
}

/* =========================
   Helpers: Menu checks
========================= */

type MenuMatch = { ok: true; canonical: string } | { ok: false; reason: string; suggestions: string[] };

function normalizeName(s: string): string {
  return s.toLowerCase().replace(/[^a-z0-9\s]/g, " ").replace(/\s+/g, " ").trim();
}

function menuMatch(itemName: string): MenuMatch {
  if (MENU_ITEMS.length === 0) {
    // no menu configured, accept anything (v1)
    return { ok: true, canonical: itemName };
  }

  const needle = normalizeName(itemName);
  if (!needle) return { ok: false, reason: "empty", suggestions: [] };

  // exact-ish contains match
  const scored = MENU_ITEMS.map((m) => {
    const hay = normalizeName(m);
    const score =
      hay === needle ? 100 :
      hay.includes(needle) ? 70 :
      needle.includes(hay) ? 60 :
      tokenOverlapScore(needle, hay);
    return { m, score };
  })
    .sort((a, b) => b.score - a.score)
    .slice(0, 5);

  if (scored[0] && scored[0].score >= 65) return { ok: true, canonical: scored[0].m };

  return {
    ok: false,
    reason: "not_on_menu",
    suggestions: scored.filter((x) => x.score >= 35).map((x) => x.m).slice(0, 3),
  };
}

function tokenOverlapScore(a: string, b: string): number {
  const A = new Set(a.split(" ").filter(Boolean));
  const B = new Set(b.split(" ").filter(Boolean));
  if (A.size === 0 || B.size === 0) return 0;
  let inter = 0;
  for (const t of A) if (B.has(t)) inter++;
  return Math.round((inter / Math.max(A.size, B.size)) * 50);
}

function loadMenuItems(): string[] {
  const csv = process.env.MENU_ITEMS?.trim();
  if (csv) return csv.split(",").map((s) => s.trim()).filter(Boolean);

  const json = process.env.MENU_JSON?.trim();
  if (json) {
    try {
      const parsed = JSON.parse(json);
      if (Array.isArray(parsed)) {
        return parsed
          .map((x) => (typeof x === "string" ? x : (x?.name ?? x?.item_name ?? "")))
          .map((s) => String(s).trim())
          .filter(Boolean);
      }
    } catch {
      // ignore
    }
  }

  return [];
}

/* =========================
   Confirmation summary
========================= */

function confirmOrderSummary(d: OrderDraft): string {
  const parts: string[] = [];

  const st = d.service_type ?? "collection";
  parts.push(st === "delivery" ? "Delivery" : "Collection");

  const t = d.pickup_time ?? "ASAP";
  parts.push(t === "ASAP" ? "ASAP" : `for ${t}`);

  if (d.customer_name) parts.push(`under ${d.customer_name}`);

  const lines = d.items
    .slice(0, 12)
    .map((it) => {
      const qty = it.quantity > 1 ? `${it.quantity}× ` : "";
      const mods = it.modifiers?.length ? ` (${it.modifiers.join(", ")})` : "";
      const note = it.special_instructions ? ` — ${it.special_instructions}` : "";
      return `${qty}${it.item_name}${mods}${note}`.trim();
    });

  const totalCount = d.items.reduce((acc, it) => acc + (it.quantity || 0), 0);

  return `Just to confirm: ${parts.join(", ")}. Items: ${lines.join("; ")}. Total items: ${totalCount}. Shall I place it?`;
}

function missingFields(d: OrderDraft): string[] {
  const missing: string[] = [];
  if (!d.service_type) missing.push("collection or delivery");
  if (!d.customer_name) missing.push("your name");
  if (!d.pickup_time) missing.push("pickup time");
  if (!d.items.length) missing.push("your order");
  if (d.service_type === "delivery" && !d.delivery_address) missing.push("delivery address");
  return missing;
}

/* =========================
   LLM Extraction Schema
========================= */

const Extracted = z.object({
  intent: z.enum(["order", "add", "remove", "change", "confirm", "cancel", "question", "restart", "unknown"]).default("unknown"),
  service_type: z.enum(["collection", "delivery"]).nullable().optional(),
  customer_name: z.string().nullable().optional(),
  pickup_time: z.string().nullable().optional(),
  delivery_address: z.string().nullable().optional(),
  notes: z.string().nullable().optional(),
  // parsed items from user's utterance
  items: z.array(z.object({
    name: z.string(),
    quantity: z.number().int().min(1).default(1),
    modifiers: z.array(z.string()).default([]),
    special_instructions: z.string().nullable().optional(),
    // for remove/change
    remove: z.boolean().optional(),
  })).default([]),
  // If user asked an FAQ, capture it so the agent can answer
  question: z.string().nullable().optional(),
});

/* =========================
   Tools
========================= */

const updateDraftTool = llm.tool({
  description:
    "Update the current order draft with extracted fields from the caller. Use this for adding/removing/changing items, name, time, service type, address, and notes. Always keep pickup_time as ASAP or HH:MM.",
  parameters: Extracted,
  execute: async (args, ctx) => {
    const agent = ctx.session.agent as OrderPilotAgent;

    // intent-based control
    if (args.intent === "restart") {
      agent.draft = newDraft();
      return { ok: true, message: "draft_reset" };
    }
    if (args.intent === "cancel") {
      agent.draft = newDraft();
      agent.draft.state = "completed";
      return { ok: true, message: "cancelled" };
    }

    // fields
    if (args.service_type) agent.draft.service_type = args.service_type;
    if (args.customer_name) agent.draft.customer_name = cleanName(args.customer_name);
    if (args.delivery_address) agent.draft.delivery_address = args.delivery_address.trim();
    if (args.notes !== undefined) agent.draft.notes = args.notes ?? null;

    if (args.pickup_time) {
      const norm = normalizePickupTime(args.pickup_time);
      if (norm) agent.draft.pickup_time = norm;
    }

    // items: add/remove/change
    if (args.items?.length) {
      for (const it of args.items) {
        const name = it.name?.trim();
        if (!name) continue;

        if (it.remove) {
          // remove first matching item
          const idx = agent.draft.items.findIndex((x) => normalizeName(x.item_name).includes(normalizeName(name)));
          if (idx >= 0) agent.draft.items.splice(idx, 1);
          continue;
        }

        // menu enforcement
        const match = menuMatch(name);
        if (!match.ok) {
          // store as a "needs clarification" note and do not add
          agent.pendingMenuClarification = {
            requested: name,
            suggestions: match.suggestions,
          };
          continue;
        }

        const canonical = match.canonical;

        // if item exists, update quantity; else add
        const existing = agent.draft.items.find((x) => normalizeName(x.item_name) === normalizeName(canonical));
        const qty = clampInt(it.quantity ?? 1, 1, 99);

        if (existing) {
          existing.quantity = qty;
          const mods = (it.modifiers ?? []).map((m) => m.trim()).filter(Boolean);
          if (mods.length) existing.modifiers = uniq([...existing.modifiers, ...mods]);
          if (it.special_instructions) existing.special_instructions = it.special_instructions;
        } else {
          agent.draft.items.push({
            item_name: canonical,
            quantity: qty,
            modifiers: uniq((it.modifiers ?? []).map((m) => m.trim()).filter(Boolean)),
            special_instructions: it.special_instructions ?? null,
          });
        }
      }
    }

    // if user intends confirm, move to confirming when ready
    if (args.intent === "confirm") agent.draft.state = "confirming";

    return { ok: true, draft: agent.draft };
  },
});

const createOrderTool = llm.tool({
  description:
    "Create a confirmed restaurant order in the OrderPilot backend. Use only after the user says yes to the confirmation.",
  parameters: z.object({
    customer_name: z.string(),
    service_type: z.enum(["collection", "delivery"]),
    pickup_time: z.string().describe('MUST be "ASAP" or "HH:MM" like "19:00".'),
    delivery_address: z.string().nullable().optional(),
    notes: z.string().nullable().optional(),
    items: z.array(
      z.object({
        item_name: z.string(),
        quantity: z.number().int().min(1),
        unit_price: z.number().nullable().optional(),
      }),
    ),
  }),
  execute: async (args) => {
    const payload = {
      restaurant_id: DEFAULT_RESTAURANT_ID,
      customer_name: cleanName(args.customer_name),
      customer_phone: null,
      pickup_time: normalizePickupTime(args.pickup_time) ?? "ASAP",
      service_type: args.service_type,
      notes: args.notes ?? null,
      delivery_address: args.delivery_address ?? null,
      items: args.items.map((i) => ({
        item_name: i.item_name,
        quantity: i.quantity,
        unit_price: i.unit_price ?? null,
      })),
    };

    // Hard guarantee format
    if (payload.pickup_time !== "ASAP" && !/^\d{2}:\d{2}$/.test(payload.pickup_time)) {
      payload.pickup_time = "ASAP";
    }

    const body = JSON.stringify(payload);

    const attempt = async () => {
      const res = await fetch(ORDERPILOT_ORDERS_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body,
      });

      const data = await res.json().catch(() => ({} as any));
      if (!res.ok) {
        throw new Error(`Order create failed ${res.status}: ${JSON.stringify(data)}`);
      }
      return data;
    };

    try {
      const data = await withTimeout(attempt(), 10_000);
      return { ok: true, order_id: data.order_id ?? data.id ?? null };
    } catch (e1: any) {
      // 1 retry
      try {
        const data = await withTimeout(attempt(), 10_000);
        return { ok: true, order_id: data.order_id ?? data.id ?? null, retried: true };
      } catch (e2: any) {
        throw new Error(`Order create failed (after retry): ${e2?.message || e2}`);
      }
    }
  },
});

const answerFaqTool = llm.tool({
  description:
    "Answer common restaurant questions (hours, address, payment, delivery vs collection, wait time). If unknown, be polite and offer to take the order anyway.",
  parameters: z.object({
    question: z.string(),
  }),
  execute: async ({ question }) => {
    const q = question.toLowerCase();

    if (q.includes("address") || q.includes("where") || q.includes("located")) {
      if (RESTAURANT_ADDRESS) return { answer: `We’re at ${RESTAURANT_ADDRESS}.` };
      return { answer: "I’m not seeing the full address here, but I can take your order now. Collection or delivery?" };
    }

    if (q.includes("open") || q.includes("close") || q.includes("opening") || q.includes("hours")) {
      if (RESTAURANT_OPENING_HOURS) return { answer: `Our hours are: ${RESTAURANT_OPENING_HOURS}.` };
      return { answer: "I don’t have the latest hours in front of me, but I can take your order now if you like." };
    }

    if (q.includes("cash") || q.includes("card") || q.includes("pay") || q.includes("payment")) {
      return { answer: "Payment depends on the shop setup — I can take the order now and staff will confirm payment method at collection." };
    }

    if (q.includes("delivery") || q.includes("deliver")) {
      return { answer: "We can do collection or delivery. Which would you like?" };
    }

    if (q.includes("how long") || q.includes("wait") || q.includes("ready") || q.includes("time")) {
      return { answer: "Usually it’s around 15–25 minutes, but it can vary. Want it ASAP or for a specific time?" };
    }

    return { answer: "I can help with that as best I can — or I can take your order now. What would you like today?" };
  },
});

/* =========================
   Agent class
========================= */

class OrderPilotAgent extends voice.Agent {
  draft: OrderDraft = newDraft();

  // if user says an item not on menu, we pause and clarify instead of blindly accepting
  pendingMenuClarification: null | { requested: string; suggestions: string[] } = null;

  constructor() {
    super({
      instructions: systemPrompt(),
      tools: [updateDraftTool, answerFaqTool, createOrderTool],
    });
  }
}

function systemPrompt(): string {
  const menuLine =
    MENU_ITEMS.length > 0
      ? `Menu is enforced. Only accept items that are on the menu list. If caller requests something not on menu, ask a quick clarification with 1–3 close suggestions, or ask them to choose another menu item.`
      : `Menu is not configured. Accept free-form items.`;

  const addressLine = RESTAURANT_ADDRESS ? `Address: ${RESTAURANT_ADDRESS}.` : `Address: unknown.`;
  const hoursLine = RESTAURANT_OPENING_HOURS ? `Opening hours: ${RESTAURANT_OPENING_HOURS}.` : `Opening hours: unknown.`;

  return `
You are OrderPilot, a premium phone order taker for ${RESTAURANT_NAME} in the UK.

GOAL
- Take orders fast and accurately.
- Sound calm, confident, warm, and human — never robotic.
- Minimal words. Ask only what you need.
- Handle interruptions, accents, background noise. If unsure, confirm once.

STATE MACHINE
- drafting: collect service type, items, name, pickup time, address (delivery only).
- confirming: give a single brief itemized summary + total count, then ask "Shall I place it?"
- placing: call create_order tool once the caller clearly says yes.
- completed: thank and end.

RULES
- Caller may give details in any order. Use update_draft tool to update fields every turn.
- Support multi-item orders with quantities and simple modifiers (e.g. "no mayo", "extra spicy").
- Allow edits: add/remove/change quantity/change time/change name/restart.
- Confirm ONLY once per order (unless caller changes something after confirmation).
- pickup_time must always be "ASAP" or "HH:MM" (e.g. "19:00"). Never send ISO timestamps.
- If caller asks a question, answer briefly using answer_faq tool if needed, then return to the order.
- If a required detail is missing, ask just one short question for the highest-priority missing field.
- If menu is enforced and item isn't on menu, do NOT accept it — ask a quick clarification.

${menuLine}
${addressLine}
${hoursLine}

CONVERSATION STYLE
- Opening: short greeting + one question.
- Use UK phrasing (e.g., "What can I get you?", "collection or delivery?", "ASAP or a time?").
- Do not ramble. 1 sentence responses when possible.

TOOLS
- update_draft: use to update draft with extracted fields and items. Use it EVERY turn.
- answer_faq: use for opening hours/address/payment/delivery/wait time.
- create_order: call only after clear confirmation. Pass items[] with item_name + quantity. Include notes if useful.

IMPORTANT
- If the user says something not on menu (and menu is enforced), ask: "We don’t have that — did you mean X, Y, or Z?"
- If still unclear, ask them to choose from the menu.
`.trim();
}

function cleanName(name: string): string {
  return name.trim().replace(/\s+/g, " ").slice(0, 64);
}

function uniq(arr: string[]): string[] {
  const out: string[] = [];
  const seen = new Set<string>();
  for (const s of arr) {
    const k = s.toLowerCase();
    if (!seen.has(k)) {
      seen.add(k);
      out.push(s);
    }
  }
  return out;
}

async function withTimeout<T>(p: Promise<T>, ms: number): Promise<T> {
  let t: any;
  const timeout = new Promise<never>((_, rej) => {
    t = setTimeout(() => rej(new Error("timeout")), ms);
  });
  try {
    return await Promise.race([p, timeout]);
  } finally {
    clearTimeout(t);
  }
}

/* =========================
   Agent Entrypoint
========================= */

defineAgent({
  entry: async (ctx: any) => {
    const session = new voice.AgentSession({
      // Voice pipeline components
      llm: LLM_MODEL,
      stt: STT_MODEL,
      tts: TTS_MODEL,
      vad: new silero.VAD(),
      // Turn detection (better endpointing on phone calls)
      // NOTE: requires model files; ensure you run download-files in your build/deploy.
      turnDetection: new livekit.turnDetector.MultilingualModel(),
      // Keep it snappy
      preemptiveGeneration: true,
      // Reduce false interruptions
      resumeFalseInterruption: true,
    });

    session.agent = new OrderPilotAgent();

    // Start the session in the room
    await session.start(ctx.room, ctx);

    // Greeting
    await session.say("Hi, thanks for calling. Collection or delivery?", {
      allowInterruptions: true,
      addToChatCtx: true,
    });

    // Runtime hooks: enforce menu clarification + keep flow tight
    session.on(voice.AgentSessionEventTypes.FunctionToolsExecuted, async () => {
      const agent = session.agent as OrderPilotAgent;

      // If menu clarification is pending, ask immediately (and do NOT progress)
      if (agent.pendingMenuClarification) {
        const { requested, suggestions } = agent.pendingMenuClarification;
        agent.pendingMenuClarification = null;

        if (suggestions.length) {
          await session.say(`We don’t have ${requested}. Did you mean ${suggestions.join(", ")}?`, {
            allowInterruptions: true,
            addToChatCtx: true,
          });
        } else {
          await session.say(`We don’t have ${requested} on the menu. What would you like instead?`, {
            allowInterruptions: true,
            addToChatCtx: true,
          });
        }
      }
    });

    // When user input is transcribed, drive state machine with short prompts
    session.on(voice.AgentSessionEventTypes.UserInputTranscribed, async (ev: any) => {
      const agent = session.agent as OrderPilotAgent;

      // If already completed, ignore further input
      if (agent.draft.state === "completed") return;

      // If the LLM moved us to confirming, and required fields are present, speak summary
      if (agent.draft.state === "confirming") {
        const missing = missingFields(agent.draft);
        if (missing.length) {
          agent.draft.state = "drafting";
          await session.say(`Quick one — what’s ${missing[0]}?`, { allowInterruptions: true, addToChatCtx: true });
          return;
        }

        await session.say(confirmOrderSummary(agent.draft), { allowInterruptions: true, addToChatCtx: true });
        return;
      }

      // If still drafting and missing fields, ask the single highest-priority missing field
      if (agent.draft.state === "drafting") {
        const missing = missingFields(agent.draft);
        if (missing.length) {
          // Keep it minimal; do not repeat if user already answered (LLM should update draft)
          await session.say(`Great — what’s ${missing[0]}?`, { allowInterruptions: true, addToChatCtx: true });
          return;
        }

        // If nothing missing, move to confirming (LLM can also do this)
        agent.draft.state = "confirming";
        await session.say(confirmOrderSummary(agent.draft), { allowInterruptions: true, addToChatCtx: true });
      }
    });

    // Silence/no input handling: reprompt once, then end politely
    let reprompted = false;
    session.on(voice.AgentSessionEventTypes.MetricsCollected, async (m: any) => {
      // Heuristic: if we see extended idle and haven't reprompted, do it
      const idleMs = m?.vad?.idleTimeMs ?? m?.idleTimeMs ?? null;
      if (typeof idleMs === "number" && idleMs > 18_000) {
        if (!reprompted) {
          reprompted = true;
          await session.say("Sorry—are you still there? What would you like today?", { allowInterruptions: true, addToChatCtx: true });
        } else {
          await session.say("No problem. Please call again when you’re ready. Bye.", { allowInterruptions: false, addToChatCtx: false });
          // Best-effort close: stop the session
          try {
            await session.close();
          } catch {
            // ignore
          }
        }
      }
    });
  },
});

/* =========================
   Run as app (Local/Worker)
========================= */

cli.runApp(new ServerOptions({ agent: fileURLToPath(import.meta.url) }));
