'use strict';
// VODER_GLM_REQUIRED_SETTINGS_V1: keep setup output compatible with model-catalog.
function applyRequiredModelSettings(item){
 if(String(item.id).toLowerCase()!=='glm-5.3-flash')return item;
 item.reasoning=true;
 if(!['low','high','max'].includes(item.thinkingLevel))item.thinkingLevel='low';
 item.compat={...(item.compat||{}),thinkingFormat:'zai',supportsReasoningEffort:true};
 return item;
}
const fs=require('node:fs/promises'),path=require('node:path'),os=require('node:os');
const {randomUUID}=require('node:crypto');
const invalid=message=>Object.assign(Error(message),{code:'VISION_CONFIGURATION_INVALID'});
async function privateDirectory(dir){
 const parents=[];for(let p=dir;;p=path.dirname(p)){parents.push(p);if(path.dirname(p)===p)break;}
 for(const p of parents.reverse()){
  try{const st=await fs.lstat(p);if(!st.isDirectory()||st.isSymbolicLink())throw invalid('Configuration path must contain regular directories');}
  catch(e){if(e.code==='ENOENT')await fs.mkdir(p,{mode:0o700});else throw e;}
 }
}
async function saveVisionModel(model,{base=path.join(os.homedir(),'.config/voder')}={}){
 if(!model||!/^[-\w./:+]{1,128}$/.test(model.id)||typeof model.apiKey!=='string'||model.apiKey.length<8||model.apiKey.length>4096||/[\r\n\0]/.test(model.apiKey))throw invalid('Invalid model identity or credential');
 const url=new URL(model.baseUrl);
 if(url.protocol!=='https:'||url.username||url.password||url.search||url.hash)throw invalid('Use an HTTPS API base URL');
 if(!['openai-completions','openai-responses','anthropic-messages','google-generative-ai'].includes(model.api))throw invalid('Unsupported model protocol');
 if(!Number.isSafeInteger(model.contextWindow)||model.contextWindow<32768||model.contextWindow>1048576)throw invalid('Use the provider documented context size');
 await privateDirectory(base);const dest=path.join(base,'additional-models.json');let previous=null,catalog={version:1,models:[]};
 try{const st=await fs.lstat(dest);if(!st.isFile()||st.isSymbolicLink()||st.mode&0o077||st.uid!==process.getuid())throw invalid('Existing model catalog has unsafe permissions');previous=await fs.readFile(dest);catalog=JSON.parse(previous);}
 catch(e){if(e.code!=='ENOENT')throw e;}
 if(catalog.version!==1||!Array.isArray(catalog.models)||Object.keys(catalog).some(k=>!['version','models'].includes(k)))throw invalid('Existing model catalog is invalid');
 const others=catalog.models.filter(x=>x.provider!=='voder-secondary-vision');
 const visionItem={provider:'voder-secondary-vision',id:model.id,api:model.api,baseUrl:url.href.replace(/\/$/,''),apiKey:model.apiKey,contextWindow:model.contextWindow,maxTokens:Math.min(8192,model.contextWindow),reasoning:false,thinkingLevel:'off',input:['text','image'],compat:{}};
 applyRequiredModelSettings(visionItem);
 const next={version:1,models:[...others,visionItem]};
 const temp=path.join(base,'.vision-'+randomUUID());
 try{
  await fs.writeFile(temp,JSON.stringify(next,null,2)+'\n',{mode:0o600,flag:'wx'});
  let current=null;try{current=await fs.readFile(dest);}catch(e){if(e.code!=='ENOENT')throw e;}
  if(previous===null?current!==null:current===null||!previous.equals(current))throw invalid('Model settings changed during setup; retry');
  await fs.rename(temp,dest);
 }finally{await fs.rm(temp,{force:true});}
 return {status:'VISION_CONFIGURED_NOT_YET_LIVE_VERIFIED',model:model.id};
}
async function configureVision(vscode){
 const id=await vscode.window.showInputBox({title:'Voder · Vision model',prompt:'Exact ID of a model your provider documents as supporting images',ignoreFocusOut:true});if(id===undefined)return;
 const baseUrl=await vscode.window.showInputBox({title:'Voder · Vision API base URL',prompt:'HTTPS base URL from your provider documentation',ignoreFocusOut:true});if(baseUrl===undefined)return;
 const api=await vscode.window.showQuickPick(['openai-completions','openai-responses','anthropic-messages','google-generative-ai'],{title:'Voder · Vision API protocol'});if(!api)return;
 const apiKey=await vscode.window.showInputBox({title:'Voder · Vision API key',password:true,prompt:'Saved privately; never included in model prompts or evidence exports',ignoreFocusOut:true});if(apiKey===undefined)return;
 const context=await vscode.window.showInputBox({title:'Voder · Vision context capacity',prompt:'Documented token capacity for this exact model, 32768–1048576',ignoreFocusOut:true});if(context===undefined)return;
 const result=await saveVisionModel({id:id.trim(),baseUrl:baseUrl.trim(),api,apiKey,contextWindow:Number(context)});
 await vscode.window.showInformationMessage('Vision model configured. The next image task will use it; live image recognition still needs verification.');
 return result;
}
async function inputDiagnostics(vscode,extensionRoot,agentRoot){
 const {createHash}=require('node:crypto');const hash=bytes=>createHash('sha256').update(bytes).digest('hex');
 const config=require('./model-env.cjs').modelEnv();let secondary=[];
 const file=path.join(os.homedir(),'.config/voder/additional-models.json');
 try{const st=await fs.lstat(file);if(!st.isFile()||st.isSymbolicLink()||st.mode&0o077||st.uid!==process.getuid())throw invalid('Unsafe model catalog');const c=JSON.parse(await fs.readFile(file,'utf8'));secondary=(c.models||[]).map(x=>({provider:x.provider,id:x.id,input:x.input||['text']}));}catch(e){if(e.code!=='ENOENT')throw e;}
 const metadata=JSON.parse(await fs.readFile(path.join(extensionRoot,'package.json'),'utf8'));
 const sources={};for(const name of ['reference-files.cjs','pdf-vision.cjs','document-input.cjs'])sources[name]=hash(await fs.readFile(path.join(extensionRoot,'src',name)));
 const report={active_extension_version:metadata.version,active_extension_path:extensionRoot,extension_host_pid:process.pid,agent_root:agentRoot,primary:{id:config.VODER_MODEL_ID,input:config.VODER_MODEL_INPUT||'text'},secondary,sources,visual_model_understanding:'REQUIRES_REAL_CHALLENGE',model_calls:0};
 const document=await vscode.workspace.openTextDocument({language:'json',content:JSON.stringify(report,null,2)});await vscode.window.showTextDocument(document,{preview:true});return report;
}
module.exports={saveVisionModel,configureVision,inputDiagnostics};
