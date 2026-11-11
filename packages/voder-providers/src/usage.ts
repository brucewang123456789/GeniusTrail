import type {TokenUsage} from "./contracts.js";
/** Prices must be explicitly configured; no invented default model tariffs. */
export type PricePerMillion = Readonly<{inputUsd: number; outputUsd: number}>;
export type UsageSnapshot = Readonly<{requests: number; reported: number; inputTokens: number; outputTokens: number; costUsd?: number}>;
export class UsageLedger {
  #requests = 0;
  #reported = 0;
  #input = 0;
  #output = 0;
  #usd = 0;
  #allPriced = true;
  record(usage: TokenUsage | undefined, price?: PricePerMillion): void {
    if (!usage) { this.#requests++; this.#allPriced = false; return; }
    if (!Number.isSafeInteger(usage.inputTokens) || !Number.isSafeInteger(usage.outputTokens) || usage.inputTokens < 0 || usage.outputTokens < 0) throw new TypeError("invalid usage");
    if (price && (!Number.isFinite(price.inputUsd) || price.inputUsd < 0 || !Number.isFinite(price.outputUsd) || price.outputUsd < 0)) throw new TypeError("invalid price");
    if (!Number.isSafeInteger(this.#input + usage.inputTokens) || !Number.isSafeInteger(this.#output + usage.outputTokens)) throw new RangeError("usage overflow");
    this.#requests++;
    this.#reported++;
    this.#input += usage.inputTokens;
    this.#output += usage.outputTokens;
    if (price) this.#usd += (usage.inputTokens * price.inputUsd + usage.outputTokens * price.outputUsd) / 1_000_000;
    else this.#allPriced = false;
  }
  snapshot(): UsageSnapshot {
    return Object.freeze({requests: this.#requests, reported: this.#reported, inputTokens: this.#input, outputTokens: this.#output, ...(this.#allPriced ? {costUsd: this.#usd} : {})});
  }
}
