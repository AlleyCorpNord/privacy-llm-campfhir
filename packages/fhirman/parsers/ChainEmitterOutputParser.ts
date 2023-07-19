import { NoOpOutputParser } from "langchain/output_parser/noop";
// @ts-ignore
import {
  ModelOutputEmitter,
  MODEL_OUTPUT_EVENT,
} from "../events/ModelOutputEmitter.ts";
import { SessionLogger } from "../helpers/sessionLogger.ts";

export class ChainEmitterOutputParser extends NoOpOutputParser {
  chain_name: string;
  emitter: ModelOutputEmitter;

  constructor(chain_name: string, emitter: ModelOutputEmitter) {
    super();
    this.chain_name = chain_name;
    this.emitter = emitter;
  }

  async parse(text: string) {
    try {
      const output = await super.parse(text);

      this.log_chain_name();

      output
        .split("\n")
        .filter((line) => line)
        .forEach((line) => {
          this.emit(line, this.chain_name);
          this.log(this.agentStepLine(line));
        });
      this.log("\n");

      return output;
    } catch (error) {
      this.log("LLM Chain error: ", error);
      this.log("LLM Chain text: ", text);
      this.log("LLM Chain text type: ", typeof text);
      throw error;
    }
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
