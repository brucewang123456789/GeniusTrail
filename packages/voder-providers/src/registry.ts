import type {ModelProvider} from "./contracts.js";
export class ProviderRegistry {
  readonly #providers = new Map<string, ModelProvider>();
  register(provider: ModelProvider): void {
    if (!/^[a-z][a-z0-9_-]{0,63}$/u.test(provider.id) || this.#providers.has(provider.id)) throw new TypeError("invalid or duplicate provider id");
    this.#providers.set(provider.id, provider);
  }
  get(id: string): ModelProvider {
    const provider = this.#providers.get(id);
    if (!provider) throw new RangeError("provider not registered");
    return provider;
  }
  list(): readonly string[] {return [...this.#providers.keys()].sort();}
}
