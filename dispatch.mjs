import dotenv from "dotenv";
dotenv.config({ path: ".env.local" });

import { AgentDispatchClient } from "livekit-server-sdk";

const host = process.env.LIVEKIT_HOST;
const apiKey = process.env.LIVEKIT_API_KEY;
const apiSecret = process.env.LIVEKIT_API_SECRET;
const agentName = process.env.AGENT_NAME || "orderpilot";

if (!host || !apiKey || !apiSecret) {
  console.error("Missing LIVEKIT_HOST / LIVEKIT_API_KEY / LIVEKIT_API_SECRET");
  console.error("Have:", {
    LIVEKIT_HOST: host ? "set" : "missing",
    LIVEKIT_API_KEY: apiKey ? "set" : "missing",
    LIVEKIT_API_SECRET: apiSecret ? "set" : "missing",
  });
  process.exit(1);
}

const roomName = process.argv[2] || "orderpilot-test";

const client = new AgentDispatchClient(host, apiKey, apiSecret);
const dispatch = await client.createDispatch(roomName, agentName);

console.log("Dispatched:", { roomName, agentName, dispatchId: dispatch.id });
