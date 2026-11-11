import {test} from 'node:test';
import assert from 'node:assert/strict';
import {mkdtemp,readFile,rm} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join,resolve} from 'node:path';
import {pathToFileURL} from 'node:url';
import {PiModelHost} from '../src/pi-host.mjs';
import {resolvePiImports} from '../src/pi-imports.mjs';
const {entry,aiEntry}=await resolvePiImports(process.env.VODER_TEST_PI_ORIGIN||import.meta.url);
const sdk=await import(entry),ai=await import(aiEntry);
const fixtures=process.env.VODER_ARCHIVE_FIXTURES||resolve('fixtures036');
const small=process.env.VODER_DOCUMENT_FIXTURES||resolve('public/Voder_File_Multimodal_0.1.35_Patch/fixtures');
const tool={name:'voder_authorized_action',description:'Offline only',parameters:{type:'object'},async execute(){throw Error('DENIED');}};
for(const corrupt of [false,true])test('real pinned SDK preserves prepared large and small image bytes; corruption='+corrupt,async()=>{
 const cwd=await mkdtemp(join(tmpdir(),'voder-sdk039-'));let host,runtime,seen=[],receivedSettings;
 try{
  runtime=await sdk.ModelRuntime.create({authPath:join(cwd,'auth.json'),modelsPath:join(cwd,'models.json')});
  runtime.registerProvider('offline-039',{api:'openai-completions',baseUrl:'https://example.invalid/v1',apiKey:'offline-fake-key',models:[{id:'offline',name:'Offline',reasoning:false,input:['text','image'],cost:{input:0,output:0,cacheRead:0,cacheWrite:0},contextWindow:1048576,maxTokens:1024}]});
  const model=runtime.getModel('offline-039','offline');assert.ok(model);
  const inherited={images:{autoResize:true,blockImages:false},retry:{enabled:false},quietStartup:true};
  const upstream={...sdk,getAgentDir:()=>join(cwd,'agent'),normalizeContext:ai.normalizeContext,
   SettingsManager:{create:()=>sdk.SettingsManager.inMemory(inherited),inMemory:settings=>sdk.SettingsManager.inMemory(settings)},
   async createAgentSession(options){receivedSettings=options.settingsManager;const result=await sdk.createAgentSession({...options,agentDir:join(cwd,'agent'),model,modelRuntime:runtime});
    result.session.agent.streamFunction=(m,c)=>{seen.push(c.messages.flatMap(x=>Array.isArray(x.content)?x.content.filter(b=>b.type==='image'):[]));throw Error('OFFLINE_CAPTURE_STOP');};
    if(corrupt){const prompt=result.session.prompt.bind(result.session);result.session.prompt=(goal,options)=>prompt(goal,{...options,images:options.images.map((x,i)=>i?x:{...x,data:options.images[1].data})});}
    return result;
   }};
  const images=await Promise.all([join(fixtures,'wide.png'),join(small,'embedded.png')].map(async p=>({type:'image',mimeType:'image/png',data:(await readFile(p)).toString('base64')})));
  host=await new PiModelHost({upstream,modelProfiles:[model],maxModelRequests:1}).createSession({cwd,systemPrompt:'Offline capture only',tools:[tool]});
  let error;try{await host.prompt('Read both images',{images});}catch(e){error=e;}
  assert.equal(receivedSettings.getImageAutoResize(),false);assert.equal(receivedSettings.getSettings().retry.enabled,false);assert.equal(inherited.images.autoResize,true);
  if(corrupt){assert.equal(error?.code,'REFERENCE_IMAGE_CHANGED');assert.equal(seen.length,0);}
  else{assert.notEqual(error?.code,'REFERENCE_IMAGE_CHANGED');assert.equal(seen.length,1);assert.deepEqual(seen[0].map(x=>({data:x.data,mimeType:x.mimeType})),images.map(x=>({data:x.data,mimeType:x.mimeType})));}
 }finally{host?.dispose();runtime?.dispose?.();await rm(cwd,{recursive:true,force:true});}
});
