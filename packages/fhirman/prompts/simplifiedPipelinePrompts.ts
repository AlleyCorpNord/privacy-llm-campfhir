import { PromptTemplate } from "https://esm.sh/langchain/prompts";

const INSTRUCTIONS_RETRIEVE_FHIR_CALL = `** INSTRUCTIONS **
The user will submit a question in plain English related to an electronic health records database.
Your task is to output a valid JSON string query based on the user input.
The format of the JSON string query should be: {{"endpoint": <ENDPOINT>, "params": {{"<PARAMETER code>": "<PARAMETER value>"}}}}.
Follow this format exactly, only replacing the values in triangular brackets ("<" and ">") as appropriate.'
### Task:
Question: {input}
JSON string query:
`;

const INSTRUCTION_SUMMARIZE_FHIR_RESULTS = `** INSTRUCTIONS **
The user will submit a question in plain English related to electronic health records database.
Data from the electronic heath records database have been retrieved previously.
Use the data provided to answer the question.
The data are provided in a JSON format below:
{fhir_results}
Question: {input}

Think before answering.  Only use the data provided for the answer.
If the data don't have the answers, then say "I don't know".`;

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
