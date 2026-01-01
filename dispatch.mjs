import dotenv from "dotenv";
dotenv.config({ path: ".env.local" });

import { AgentDispatchClient } from "livekit-server-sdk";

/**
 * REQUIRED ENV VARS
 */
const LIVEKIT_HOST = process.env.LIVEKIT_HOST; // e.g. https://xxxx.livekit.cloud
const LIVEKIT_API_KEY = process.env.LIVEKIT_API_KEY;
const LIVEKIT_API_SECRET = process.env.LIVEKIT_API_SECRET;

/**
 * MUST MATCH defineAgent({ name: "orderpilot-phone-agent" })
 */
const AGENT_NAME =
  process.env.AGENT_NAME || "orderpilot-phone-agent";

/**
 * Room to dispatch into
 */
const ROOM_NAME = process.argv[2] || "orderpilot-test";

/**
 * Validation
 */
if (!LIVEKIT_HOST || !LIVEKIT_API_KEY || !LIVEKIT_API_SECRET) {
  console.error("❌ Missing LiveKit credentials");
  console.error({
    LIVEKIT_HOST: LIVEKIT_HOST ? "set" : "missing",
    LIVEKIT_API_KEY: LIVEKIT_API_KEY ? "set" : "missing",
    LIVEKIT_API_SECRET: LIVEKIT_API_SECRET ? "set" : "missing",
  });
  process.exit(1);
}

(async () => {
  try {
    const client = new AgentDispatchClient(
      LIVEKIT_HOST,
      LIVEKIT_API_KEY,
      LIVEKIT_API_SECRET
    );

    const dispatch = await client.createDispatch(
      ROOM_NAME,
      AGENT_NAME
    );

    console.log("✅ Dispatched agent successfully:");
    console.log({
      roomName: ROOM_NAME,
      agentName: AGENT_NAME,
      dispatchId: dispatch.id,
    });
  } catch (err) {
    console.error("❌ Failed to dispatch agent");
    console.error(err);
    process.exit(1);
  }
})();
