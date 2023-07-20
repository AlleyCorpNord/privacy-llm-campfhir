import { BaseLLMParams, LLM } from "langchain/llms/base";

export interface TextGenerationInferenceInput {
  /** Sampling temperature to use */
  temperature?: number;

  /**
   * Maximum number of tokens to generate in the completion.
   */
  maxTokens?: number;

  // /** Model to use */
  // model?: string;

  // apiKey?: string;

  /** Model service URI */
  model_service_uri?: string;
}

export class TextGenerationInferenceLLM
  extends LLM
  implements TextGenerationInferenceInput
{
  model_service_uri: string;
  temperature: number;
  maxTokens: number;

  constructor(fields?: Partial<TextGenerationInferenceInput> & BaseLLMParams) {
    super(fields ?? {});
    this.maxTokens = fields?.maxTokens ?? 200;
    this.temperature = fields?.temperature ?? 0.01;

    this.model_service_uri =
      fields?.model_service_uri ??
      "http://host.docker.internal:5000/invocations";
  }

  _llmType() {
    return "text-generation-inference";
  }

  /** @ignore */
  async _call(
    prompt: string,
    options: this["ParsedCallOptions"]
  ): Promise<string> {
    console.log("TGI prompt:", prompt);
    const response = await fetch(this.model_service_uri, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        inputs: prompt,
        parameters: {
          temperature: this.temperature,
          max_new_tokens: this.maxTokens,
        },
      }),
    })
      .then((response) => response.json())
      .then((response) => response.generated_text);
    console.log("TGI response:", response);
    return response;
  }
}
