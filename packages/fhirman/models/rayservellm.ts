import { BaseLLMParams, LLM } from "langchain/llms/base";

export interface RayServeLLMInput {
  model_service_uri?: string;
}

export class RayServeLLM extends LLM implements RayServeLLMInput {
  model_service_uri: string;

  constructor(fields?: Partial<RayServeLLMInput> & BaseLLMParams) {
    super(fields ?? {});

    this.model_service_uri =
      fields?.model_service_uri ??
      "http://host.docker.internal:8000";
  }

  _llmType() {
    return "RayServe";
  }

  /** @ignore */
  async _call(
    prompt: string,
    options: this["ParsedCallOptions"]
  ): Promise<string> {
    console.log("RayServe prompt:", prompt);
    const response = await fetch(this.model_service_uri, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        inputs: prompt,
      }),
    }).then((response) => response.text());

    console.log("RayServe response:", response);
    return response;
  }
}
