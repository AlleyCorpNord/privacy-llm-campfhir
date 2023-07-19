import { AgentExecutor, ZeroShotAgent } from "langchain/agents";
import { LLMChain, SequentialChain, TransformChain } from "langchain/chains";
import { BufferMemory } from "langchain/memory";

import { getCurrentUser } from "../helpers/currentUser.ts";
import { MlflowLLM } from "../models/mlflowllm.ts";
import { createOpenAIInstance } from "../models/openai.ts";

import { ChainEmitterOutputParser } from "../parsers/ChainEmitterOutputParser.ts";
import { EmitterOutputParser } from "../parsers/EmitterOutputParser.ts";
import { retrieve_fhir_request_prompt, summarize_fhir_results_prompt } from "../prompts/simplifiedPipelinePrompts.ts";
import { assistantPrompt } from "../prompts/assistantPrompt.ts";
import { call_fhir_server } from "../tools/FhirAPIServer.ts";
import { FhirQuestion } from "../tools/FhirQuestion.ts";

// @ts-ignore
import { ModelOutputEmitter } from "../events/ModelOutputEmitter.ts";

import process from "process";

export type AssistantAgent = {
  events: ModelOutputEmitter;
  agent: AgentExecutor | SequentialChain;
};

export async function createAssistantAgent(): Promise<AssistantAgent> {
  const outputEmitter = new ModelOutputEmitter();
  const currentUser = await getCurrentUser();

  const tools = [new FhirQuestion(outputEmitter)];

  const agentPrompt = await assistantPrompt(currentUser, tools);

  console.log(
    "OPENAI_API_KEY is up ",
    process.env.OPENAI_API_KEY ? true : false
  );
  const llm = process.env.OPENAI_API_KEY
    ? createOpenAIInstance({ temperature: 0 })
    : new MlflowLLM({
        model_service_uri: process.env.MLFLOW_LLM_MODEL_SERVICE_URI,
      });
  const memory = new BufferMemory({ memoryKey: "chat_history" });
  const llmChain = new LLMChain({
    llm,
    prompt: agentPrompt,
  });

  const agent = new ZeroShotAgent({
    llmChain,
    allowedTools: tools.map((tool) => tool.name),
    outputParser: new EmitterOutputParser("FHIR Assistant", outputEmitter),
  });

  const executor = AgentExecutor.fromAgentAndTools({
    agent,
    tools,
    memory,
  });

  return {
    agent: executor,
    events: outputEmitter,
  };
}

export async function createSequentialChain(): Promise<AssistantAgent> {
  const outputEmitter = new ModelOutputEmitter();
  const retrieve_data_prompt = retrieve_fhir_request_prompt();
  const summarize_data_prompt = summarize_fhir_results_prompt();

  const retrieve_fhir_call_llm = new MlflowLLM({
    model_service_uri: process.env.MLFLOW_LLM_MODEL_SERVICE_URI,
  });

  const retrieve_fhir_call_llmChain = new LLMChain({
    retrieve_fhir_call_llm,
    prompt: retrieve_data_prompt,
    outputParser: new ChainEmitterOutputParser(
      "FHIR Call Retriever",
      outputEmitter
    ),
    outputKey: "query",
  });

  const call_fhir_server_tfm = new TransformChain({inputVariables: ["query"], outputVariables: ["fhir_results"], transform: call_fhir_server});


  const summarize_fhir_results_llmChain = new LLMChain({
    retrieve_fhir_call_llm,
    prompt: summarize_data_prompt,
    outputParser: new ChainEmitterOutputParser(
      "FHIR Results Summarizer",
      outputEmitter
    ),
    outputKey: "answer",
  });

  
  const llm_pipeline = new SequentialChain({
    chains: [
      retrieve_fhir_call_llmChain,
      call_fhir_server_tfm,
      summarize_fhir_results_llmChain,
    ],
    inputVariables: ["input"],
    outputVariables: ["answer"],
  });

  return {
    agent: llm_pipeline,
    events: outputEmitter,
  };
}
