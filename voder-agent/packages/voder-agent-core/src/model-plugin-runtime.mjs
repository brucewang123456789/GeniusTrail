import {resolveModelLimits} from './model-limits.mjs';
import {readAdditionalModels} from './model-catalog.mjs';
/** Voder-controlled provider selection. Pi is used as model runtime, NEVER as authority for tools/VGR. */
import {readFile} from 'node:fs/promises';
import {resolve,dirname,join} from 'node:path';
import {fileURLToPath,pathToFileURL} from 'node:url';
import {tmpdir} from 'node:os';
import {randomUUID} from 'node:crypto';
const root=resolve(dirname(fileURLToPath(import.meta.url)),'../../..');
const modes=Object.freeze({api:'voder-provider-cloud',weights:'voder-provider-open'});
export async function readInstalledMode(){
 let raw;
 try{raw=await readFile(join(root,'config/model-provider-active.json'),'utf8');}
 catch(e){if(e.code==='ENOENT')return null;throw e;}
 let cfg;
 try{cfg=JSON.parse(raw);}catch{throw Error('CORRUPT_MODEL_PLUGIN_SELECTION');}
 if(cfg.version!==1 || !Object.hasOwn(modes,cfg.mode)||Object.keys(cfg).some(k=>!['version','mode'].includes(k)))
   throw Error('INVALID_MODEL_PLUGIN_SELECTION');
 return cfg.mode;
}
/** Returns null only when absent AND there is no partial model credential configuration. */
export async function bindInstalledModel(api,env=process.env,profiles=[],{catalogFile}={}){
 const selected=await readInstalledMode();
 if(!selected){
   if(env.VODER_MODEL_ID || env.VODER_MODEL_API_KEY) throw Error('INSTALL_MODEL_PLUGIN_FIRST');
   return null;
 }
 const primaryConfigured=Boolean(env.VODER_MODEL_ID&&env.VODER_MODEL_API_KEY);
 if(Boolean(env.VODER_MODEL_ID)!==Boolean(env.VODER_MODEL_API_KEY))throw Error('MODEL_ID_AND_API_KEY_REQUIRED');
 const additional=selected==='api'?await readAdditionalModels({env,...(catalogFile?{file:catalogFile}:{})}):[];
 if(!primaryConfigured&&!additional.length)return null;
 if(typeof api.ModelRuntime?.create!=='function')throw Error('INCOMPATIBLE_PI_MODEL_RUNTIME');
 if(!Array.isArray(profiles)||profiles.some(x=>!x||Object.keys(x).some(k=>!['id','contextWindow','maxTokens'].includes(k))||typeof x.id!=='string'||!x.id||x.id.length>128||!Number.isSafeInteger(x.contextWindow)||x.contextWindow<1024||x.contextWindow>1048576||!Number.isSafeInteger(x.maxTokens)||x.maxTokens<64||x.maxTokens>x.contextWindow)||new Set(profiles.map(x=>x.id)).size!==profiles.length||(!primaryConfigured&&profiles.length))throw Error('INVALID_MODEL_PROFILES');
 let p=null,limits=null;
 if(primaryConfigured){
   const file=join(root,'plugins',modes[selected],'plugin.mjs');
   const plugin=await import(pathToFileURL(file).href);
   if(plugin.mode!==selected)throw Error('MODEL_PLUGIN_MODE_MISMATCH');
   p=plugin.resolveConfiguration(env);
   if(profiles.some(x=>x.id===p.model))throw Error('INVALID_MODEL_PROFILES');
 }
 const modelRuntime=await api.ModelRuntime.create({modelsPath:null,authPath:join(tmpdir(),'voder-pi-auth-absent-'+randomUUID()+'.json'),allowModelNetwork:false,refreshOnCreate:false});
 let declarations=[];
 if(p){
   limits=await resolveModelLimits({configuration:p,env,runtime:modelRuntime});
   const documentedVision=p.model==='deepseek-flash'&&/^https:\/\/api\.deepseek\.com(?:\/|$)/.test(p.baseUrl||'');
   const inputDeclaration=env.VODER_MODEL_INPUT??(documentedVision?'text,image':'text');
   if(!['text','text,image'].includes(inputDeclaration))throw Error('INVALID_MODEL_INPUT_CAPABILITY');
   const primaryInput=Object.freeze(inputDeclaration==='text,image'?['text','image']:['text']);
   declarations=[{id:p.model,contextWindow:limits.contextWindow,maxTokens:limits.maxTokens},...profiles];
   modelRuntime.registerProvider(p.provider,{
    name:`Voder ${selected} provider`,baseUrl:p.baseUrl,api:p.api,apiKey:p.apiKey,
    models:declarations.map(x=>({...x,name:x.id,api:p.api,reasoning:false,input:primaryInput,cost:{input:0,output:0,cacheRead:0,cacheWrite:0}})),
   });
 }
 for(const x of additional){
   // A provider's declared models can be extended by configuration, not by model output.
   if(additional.some(y=>y!==x&&y.provider===x.provider))throw Error('SECONDARY_PROVIDER_MUST_HAVE_ONE_EXPLICIT_MODEL');
   modelRuntime.registerProvider(x.provider,{name:'Voder secondary provider',baseUrl:x.baseUrl,api:x.api,apiKey:x.apiKey,
    models:[{id:x.id,name:x.id,api:x.api,contextWindow:x.contextWindow,maxTokens:x.maxTokens,reasoning:x.reasoning,input:x.input,compat:x.compat,
      ...(x.id.toLowerCase()==='glm-5.3-flash'?{thinkingLevelMap:{off:null,minimal:null,low:'low',medium:null,high:'high',xhigh:null,max:'max'}}:x.thinkingLevelMap?{thinkingLevelMap:x.thinkingLevelMap}:{}),cost:{input:0,output:0,cacheRead:0,cacheWrite:0}}]});
 }
 const scopedModels=[...declarations.map(x=>({model:modelRuntime.getModel(p.provider,x.id),thinkingLevel:'off'})),...additional.map(x=>({model:modelRuntime.getModel(x.provider,x.id),thinkingLevel:x.thinkingLevel}))];
 const expectedIds=[...declarations.map(x=>({provider:p.provider,id:x.id})),...additional];
 if(scopedModels.some((x,i)=>!x.model||x.model.provider!==expectedIds[i].provider||x.model.id!==expectedIds[i].id))throw Error('MODEL_REGISTRATION_FAILED');
 const model=scopedModels[0]?.model;
 if(!model)throw Error('MODEL_REGISTRATION_FAILED');
 if(!limits)limits=Object.freeze({source:'CONFIGURED_MODEL_CATALOG',capability_verified:false,contextWindow:model.contextWindow,maxTokens:model.maxTokens});
 return Object.freeze({mode:selected,modelRuntime,model,scopedModels,limits,policies:scopedModels.map(x=>Object.freeze({provider:x.model.provider,id:x.model.id,thinkingLevel:x.thinkingLevel}))});
}
