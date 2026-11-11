/** Provider-neutral plug-in contract. VGR admission and delivery always remain in Rust Core. */
import type {SandboxTask,Receipt} from './contract.ts';
export interface SandboxPlugin {
  readonly id: string;
  readonly version: string;
  execute(task:SandboxTask):Promise<Receipt>;
}
export type PluginFactory = () => Promise<SandboxPlugin>;
const ID = /^[a-z][a-z0-9-]{1,39}$/;
/** Allowlist, not arbitrary path-based code loading or shell plugins. */
export class PluginRegistry {
  readonly #factories=new Map<string,PluginFactory>();
  register(id:string, factory:PluginFactory):void {
    if(!ID.test(id)||this.#factories.has(id)||typeof factory!=='function')throw new Error('INVALID_OR_DUPLICATE_SANDBOX_PLUGIN');
    this.#factories.set(id,factory);
  }
  list():string[]{return [...this.#factories.keys()].sort();}
  async resolve(id:string):Promise<SandboxPlugin>{
    const f=this.#factories.get(id);if(!f)throw new Error(`SANDBOX_PLUGIN_NOT_INSTALLED:${id}`);
    const instance=await f();if(instance.id!==id||typeof instance.execute!=='function')throw new Error('PLUGIN_CONTRACT_VIOLATION');
    return instance;
  }
}
