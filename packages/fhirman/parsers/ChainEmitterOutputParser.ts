import { BaseOutputParser } from "langchain/schema/output_parser";
// @ts-ignore
import {
  ModelOutputEmitter,
  MODEL_OUTPUT_EVENT,
} from "../events/ModelOutputEmitter.ts";
import { SessionLogger } from "../helpers/sessionLogger.ts";

export class ChainEmitterOutputParser extends BaseOutputParser<string> {
  lc_namespace = ["langchain", "output_parsers", "default"];

  lc_serializable = true;
  chain_name: string;
  emitter: ModelOutputEmitter;

  constructor(chain_name: string, emitter: ModelOutputEmitter) {
    super();
    this.chain_name = chain_name;
    this.emitter = emitter;
  }

  async parse(text: string): Promise<string> {
    try {
      this.log_chain_name();

      text
        .split("\n")
        .filter((line) => line)
        .forEach((line) => {
          this.emit(line, this.chain_name);
          this.log(this.agentStepLine(line));
        });
      this.log("\n");

      return Promise.resolve(text);
    } catch (error) {
      this.log("LLM Chain error: ", error);
      this.log("LLM Chain text: ", text);
      this.log("LLM Chain text type: ", typeof text);
      throw error;
    }
  }

  getFormatInstructions(): string {
    return "";
  }

  log(message: string, ...extra: any[]) {
    SessionLogger.log(message, ...extra);
    console.log(message, ...extra);
  }

  protected emit(message: string, tool?: string) {
    this.emitter.emit(MODEL_OUTPUT_EVENT, message, "SequentialChain", tool);
  }

  protected log_chain_name() {
    const title = `💠 ${this.chain_name} agent`;
    this.log(title);
  }

  protected agentStepLine(line: string) {
    return `⚙️ ${line.trim()}`;
  }
}
