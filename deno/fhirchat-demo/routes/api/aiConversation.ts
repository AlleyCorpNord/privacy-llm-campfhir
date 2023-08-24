import { serve } from "https://deno.land/std@0.184.0/http/server.ts";
import { type ChainValues } from "https://esm.sh/langchain/schema";
import process from "process";

import {
  createAssistantAgent,
  type AssistantAgent,
} from "$projectRoot/packages/fhirman/agents/assistant.ts";
import { MODEL_OUTPUT_EVENT } from "$projectRoot/packages/fhirman/events/ModelOutputEmitter.ts";
import { SessionLogger } from "$projectRoot/packages/fhirman/helpers/sessionLogger.ts";
import { AGENT_MOCK_RESPONSES } from "../../mocks/mock_responses.ts";
import { MOCK_THOUGHT_ACTIONS } from "../../mocks/mock_thought_actions.ts";

function handler(req: Request): Response {
  if (req.headers.get("upgrade") != "websocket") {
    return new Response(null, { status: 501 });
  }
  const { socket, response } = Deno.upgradeWebSocket(req);
  socket.addEventListener("open", () => {
    console.log("a client connected!");
  });

  let assistant: AssistantAgent;

  async function questionHandler(event: MessageEvent) {
    if (!assistant) {
      assistant = await createAssistantAgent();
      assistant.events.on(
        MODEL_OUTPUT_EVENT,
        (message: string, agentName: string, toolName: string) => {
          socket.send(
            JSON.stringify({ log: { message, agentName, toolName } })
          );
        }
      );
    }
    if (event.data === "ping") {
      socket.send("pong");
    } else if (event.data) {
      const response: ChainValues = await assistant.agent.call({
        input: event.data,
      });
      console.log("Agent response:", response);
      socket.send(JSON.stringify({ response: response.output }));
    }
  }

  function mockHandler(event: MessageEvent) {
    let index = 0;
    const interval = setInterval(() => {
      if (index >= AGENT_MOCK_RESPONSES.length) {
        clearInterval(interval);
        return;
      }

      const message = AGENT_MOCK_RESPONSES[index];
      const thought = MOCK_THOUGHT_ACTIONS[index];

      const data = JSON.stringify({
        log: {
          message,
          thought,
          agentName: "mock agent",
        },
      });

      socket.send(data);
      console.log(`Sent message: ${message}`);

      index++;
    }, 5000);

    if (event.data) {
      socket.send(
        JSON.stringify({
          response: AGENT_MOCK_RESPONSES[AGENT_MOCK_RESPONSES.length],
        })
      );
    }
  }

  if (process.env.MOCK_LLM_CALL === "true") {
    socket.addEventListener("message", mockHandler);
  } else {
    socket.addEventListener("message", questionHandler);
  }
  return response;
}

SessionLogger.init({});
serve(handler, { port: 8889 });
