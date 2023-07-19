import { LLMChain } from "langchain/chains";

import { AgentExecutor, ZeroShotAgent } from "langchain/agents";
import { BufferWindowMemory } from "langchain/memory";

import { MlflowLLM } from "../models/mlflowllm.ts";
import { createOpenAIInstance } from "../models/openai.ts";

import { EmitterOutputParser } from "../parsers/EmitterOutputParser.ts";
import { fhirQuestionPrompt } from "../prompts/fhirQuestionPrompt.ts";

import { ModelOutputEmitter } from "../events/ModelOutputEmitter.ts";
import { getCurrentUser } from "../helpers/currentUser.ts";
import { DateToolkit } from "../tools/DateToolkit.ts";
import { FhirAPIServer } from "../tools/FhirAPIServer.ts";
import { FhirDocsToolkit } from "../tools/FhirDocsToolkit.ts";

import process from "process";

export async function createFhirAgent(emitter: ModelOutputEmitter) {
  const docsToolkit = new FhirDocsToolkit();
  const dateToolkit = new DateToolkit();

  const tools = [
    ...docsToolkit.tools,
    new FhirAPIServer(),
    ...dateToolkit.tools,
  ];


  const currentUser = await getCurrentUser();
  const prompt = await fhirQuestionPrompt(currentUser, tools);

  const llm = process.env.OPENAI_API_KEY
    ? createOpenAIInstance({ modelName: "gpt-4", temperature: 0 })
    : new MlflowLLM({
        model_service_uri: process.env.MLFLOW_LLM_MODEL_SERVICE_URI,
      });

  const llmChain = new LLMChain({
    llm,
    prompt,
    // verbose: true,
  });
  const agent = new ZeroShotAgent({
    llmChain,
    allowedTools: tools.map((tool) => tool.name),
    outputParser: new EmitterOutputParser("FhirQuestion", emitter),
  });
  const memory = new BufferWindowMemory({ memoryKey: "chat_history" });
  return AgentExecutor.fromAgentAndTools({
    agent,
    tools: tools,
    memory,
  });
}
