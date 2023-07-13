import { BaseLLMParams, LLM } from "langchain/llms/base";

export interface MlflowLLMInput {
  // /** Sampling temperature to use */
  // temperature?: number;

  // /**
  //  * Maximum number of tokens to generate in the completion.
  //  */
  // maxTokens?: number;

  // /** Model to use */
  // model?: string;

  // apiKey?: string;

  /** Model service URI */
  model_service_uri?: string;
}

export class MlflowLLM extends LLM implements MlflowLLMInput {
  model_service_uri: string;

  constructor(fields?: Partial<MlflowLLMInput> & BaseLLMParams) {
    super(fields ?? {});

    this.model_service_uri =
      fields?.model_service_uri ??
      "http://host.docker.internal:5000/invocations";
  }

  _llmType() {
    return "mlflow";
  }

  /** @ignore */
  async _call(
    prompt: string,
    options: this["ParsedCallOptions"]
  ): Promise<string> {
    console.log("MLFlow prompt:", prompt);
    const response = await fetch(this.model_service_uri, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        inputs: [{ prompt: prompt }],
      }),
    })
      .then((response) => response.json())
      .then((response) => response.predictions[0]);
    console.log("MLFlow response:", response);
    return response;
  }
}
