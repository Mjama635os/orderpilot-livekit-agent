import { ServerOptions, cli, defineAgent, inference, metrics, voice, } from "@livekit/agents";
import * as livekit from "@livekit/agents-plugin-livekit";
import * as silero from "@livekit/agents-plugin-silero";
import { BackgroundVoiceCancellation } from "@livekit/noise-cancellation-node";
import dotenv from "dotenv";
import { fileURLToPath } from "node:url";
dotenv.config({ path: ".env.local" });
class OrderPilotAgent extends voice.Agent {
    constructor() {
        super({
            instructions: `
You are OrderPilot, a friendly, fast restaurant phone order-taker in the UK.
The caller is a real customer. Speak naturally like takeaway staff.

Rules:
- Keep responses short. One question at a time.
- Do not mention AI, bots, assistants, or models.
- If you didn’t hear something, say: "Sorry, could you say that again?"
- Flow (strict):
  1) Ask: collection or delivery
  2) Ask: what would you like to order
  3) Ask: name for the order
  4) Ask: time (ASAP, 7pm, in 20 minutes)
  5) If delivery: ask address + postcode
  6) Read back a short confirmation and ask yes/no
- If user says no at confirmation, ask what to change and continue.
- Avoid emojis and fancy punctuation.
`,
        });
    }
}
export default defineAgent({
    prewarm: async (proc) => {
        proc.userData.vad = await silero.VAD.load();
    },
    entry: async (ctx) => {
        // 1) Connect to the room FIRST and set the agent display name
        await ctx.connect({ name: "OrderPilot" });
        // 2) Start voice pipeline session
        const session = new voice.AgentSession({
            // STT
            stt: new inference.STT({
                model: "assemblyai/universal-streaming",
                language: "en",
            }),
            // LLM
            llm: new inference.LLM({
                model: "openai/gpt-4.1-mini",
            }),
            // TTS
            tts: new inference.TTS({
                model: "cartesia/sonic-3",
                voice: "9626c31c-bec5-4cca-baa8-f8ba9e84c8bc",
            }),
            // Turn detection + VAD
            turnDetection: new livekit.turnDetector.MultilingualModel(),
            vad: ctx.proc.userData.vad,
            voiceOptions: {
                // Generates while waiting for end of turn = snappier
                preemptiveGeneration: true,
            },
        });
        // Metrics (optional but useful)
        const usageCollector = new metrics.UsageCollector();
        session.on(voice.AgentSessionEventTypes.MetricsCollected, (ev) => {
            metrics.logMetrics(ev.metrics);
            usageCollector.collect(ev.metrics);
        });
        ctx.addShutdownCallback(async () => {
            const summary = usageCollector.getSummary();
            console.log(`Usage: ${JSON.stringify(summary)}`);
        });
        await session.start({
            agent: new OrderPilotAgent(),
            room: ctx.room,
            inputOptions: {
                // Good for browser testing; for telephony later use the telephony variant
                noiseCancellation: BackgroundVoiceCancellation(),
            },
        });
        // 3) Speak immediately so there's no awkward silence
        // This line is important: it makes it feel "human" right away.
        await session.say("Hi, thanks for calling. Is this for collection or delivery?");
    },
});
// CLI bootstrap (must be OUTSIDE defineAgent)
cli.runApp(new ServerOptions({ agent: fileURLToPath(import.meta.url) }));
//# sourceMappingURL=agent.js.map