import {test} from 'node:test';
import assert from 'node:assert/strict';
import {PluginRegistry} from '../src/plugin.ts';
import {builtinPlugins} from '../src/registry.ts';
test('modal is the default installed provider; others are not implicitly executable',()=>{
  assert.deepEqual(builtinPlugins().list(),['modal']);
  assert.throws(()=>new PluginRegistry().register('../../unsafe',async()=>null),/INVALID_OR_DUPLICATE/);
});
test('extension API admits an explicit verified provider and rejects contract mismatch',async()=>{
  const r=new PluginRegistry();r.register('alternative',async()=>({id:'alternative',version:'1.0',async execute(){return {status:'EXECUTED'};}}));
  assert.equal((await r.resolve('alternative')).id,'alternative');
  await assert.rejects(r.resolve('unknown'),/NOT_INSTALLED/);
  assert.throws(()=>r.register('alternative',async()=>null),/DUPLICATE/);
  const bad=new PluginRegistry();bad.register('configured',async()=>({id:'wrong',execute(){}}));
  await assert.rejects(bad.resolve('configured'),/CONTRACT_VIOLATION/);
});
