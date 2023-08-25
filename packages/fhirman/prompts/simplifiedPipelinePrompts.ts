import { PromptTemplate } from "https://esm.sh/langchain/prompts";

// const INSTRUCTIONS_RETRIEVE_FHIR_CALL = `** INSTRUCTIONS **
// The user will submit a question in plain English related to an electronic health records database.
// Your task is to output a valid JSON string query based on the user input.
// The format of the JSON string query should be: {{"endpoint": <ENDPOINT>, "params": {{"<PARAMETER code>": "<PARAMETER value>"}}}}.
// Follow this format exactly, only replacing the values in triangular brackets ("<" and ">") as appropriate.'
// ### Task:
// Question: {input}
// JSON string query:
// `;

const INSTRUCTIONS_RETRIEVE_FHIR_CALL = `{input}`;

const INSTRUCTION_SUMMARIZE_FHIR_RESULTS = `{input} {fhir_results}`;

export function retrieve_fhir_request_prompt() {
  return new PromptTemplate({
    template: INSTRUCTIONS_RETRIEVE_FHIR_CALL,
    inputVariables: ["input"],
  });
}

export function summarize_fhir_results_prompt() {
  return new PromptTemplate({
    template: INSTRUCTION_SUMMARIZE_FHIR_RESULTS,
    inputVariables: ["input", "fhir_results"],
  });
}
