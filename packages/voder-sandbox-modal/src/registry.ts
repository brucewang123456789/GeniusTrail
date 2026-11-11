import {PluginRegistry} from './plugin.ts';
import {createModalDriver} from './driver.ts';
import {executeModalTask} from './execute.ts';
/** A downstream distribution can register other audited providers here without changing VGR. */
export function builtinPlugins():PluginRegistry {
  const registry=new PluginRegistry();
  registry.register('modal',async()=>({
    id:'modal',version:'1.0.0',
    async execute(task){return executeModalTask(task,await createModalDriver());},
  }));
  return registry;
}

/** Operator-managed extensions are opt-in; modal is always the default installed provider. */
export {registerInstalledPlugins} from './installed.ts';
