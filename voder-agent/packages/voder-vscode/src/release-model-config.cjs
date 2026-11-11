'use strict';
/** User-initiated, extension-host-local model configuration. No model or network call. */
const fs=require('node:fs/promises');
const path=require('node:path');
const os=require('node:os');
const crypto=require('node:crypto');
const CONFIG_KEYS=['VODER_MODEL_ID','VODER_MODEL_API_KEY','VODER_MODEL_VENDOR','VODER_MODEL_PROTOCOL','VODER_MODEL_ENDPOINT'];
function fail(code){throw Object.assign(new Error(code),{code});}
async function saveModelConfig(data,{home=os.homedir()}={}){
 if(!data||Object.keys(data).some(k=>!CONFIG_KEYS.includes(k))||
  typeof data.VODER_MODEL_ID!=='string'||!/^[-\w./:+]{1,128}$/.test(data.VODER_MODEL_ID)||
  typeof data.VODER_MODEL_API_KEY!=='string'||data.VODER_MODEL_API_KEY.length<8||data.VODER_MODEL_API_KEY.length>4096||
  Object.values(data).some(v=>typeof v!=='string'||/[\r\n\0]/.test(v)||v.length>4096))fail('MODEL_CONFIG_INVALID');
 if(data.VODER_MODEL_ENDPOINT){let url;try{url=new URL(data.VODER_MODEL_ENDPOINT);}catch{fail('MODEL_ENDPOINT_INVALID');}
  if(url.protocol!=='https:'&&!(url.protocol==='http:'&&['127.0.0.1','localhost','[::1]'].includes(url.hostname))||url.username||url.password||url.search||url.hash)fail('MODEL_ENDPOINT_UNSAFE');}
 const parent=path.join(home,'.config'),directory=path.join(parent,'voder');
 await fs.mkdir(parent,{recursive:true,mode:0o700});
 const ps=await fs.lstat(parent);
 if(!ps.isDirectory()||ps.isSymbolicLink())fail('MODEL_CONFIG_PARENT_UNSAFE');
 await fs.mkdir(directory,{mode:0o700,recursive:true});
 const ds=await fs.lstat(directory);
 if(!ds.isDirectory()||ds.isSymbolicLink()||(ds.mode&0o077)!==0||(process.getuid&&ds.uid!==process.getuid()))fail('MODEL_CONFIG_DIR_UNSAFE');
 const dest=path.join(directory,'model-provider.json');
 let existing=false;
 try{const s=await fs.lstat(dest);existing=true;
  if(!s.isFile()||s.isSymbolicLink()||(s.mode&0o077)!==0||(process.getuid&&s.uid!==process.getuid()))fail('EXISTING_MODEL_CONFIG_UNSAFE');
 }catch(e){if(e.code!=='ENOENT')throw e;}
 const temp=path.join(directory,'.model-provider-'+crypto.randomUUID());
 const f=await fs.open(temp,'wx',0o600);
 try{await f.writeFile(JSON.stringify(data)+'\n');await f.sync();}finally{await f.close();}
 try{await fs.rename(temp,dest);}catch(e){await fs.rm(temp,{force:true});throw e;}
 return {path:dest,updated:existing};
}
async function configureModel(vscode){
 if(process.platform!=='linux'){
  await vscode.window.showErrorMessage('This release runs natively on Linux x64. On Windows/macOS, connect to a Linux x64 host with VS Code Remote SSH and configure Voder there.');return false;
 }
 const preset=await vscode.window.showQuickPick([
  {label:'DeepSeek',id:'deepseek-flash',vendor:'deepseek',api:'openai-completions',endpoint:'https://api.deepseek.com/v1'},
  {label:'OpenAI',id:'',vendor:'openai',api:'openai-responses',endpoint:'https://api.openai.com/v1'},
  {label:'Anthropic',id:'',vendor:'anthropic',api:'anthropic-messages',endpoint:'https://api.anthropic.com'},
  {label:'Google Gemini',id:'',vendor:'gemini',api:'openai-completions',endpoint:'https://generativelanguage.googleapis.com/v1beta/openai'},
  {label:'Kimi',id:'',vendor:'kimi',api:'openai-completions',endpoint:'https://api.moonshot.ai/v1'},
  {label:'GLM',id:'',vendor:'glm',api:'openai-completions',endpoint:'https://open.bigmodel.cn/api/paas/v4'},
  {label:'Grok',id:'',vendor:'grok',api:'openai-completions',endpoint:'https://api.x.ai/v1'},
  {label:'Qwen',id:'',vendor:'qwen',api:'openai-completions',endpoint:'https://dashscope.aliyuncs.com/compatible-mode/v1'},
  {label:'OpenRouter',id:'',vendor:'openrouter',api:'openai-completions',endpoint:'https://openrouter.ai/api/v1'},
  {label:'Custom OpenAI-compatible API',id:'',vendor:'custom',api:'openai-completions',endpoint:''}
 ],{title:'Voder — Model Provider',placeHolder:'Credentials remain on this VS Code extension host'});
 if(!preset)return false;
 const model=await vscode.window.showInputBox({title:'Model identifier',value:preset.id,prompt:'Enter the model identifier configured with your API provider',validateInput:v=>/^[-\w./:+]{1,128}$/.test(v)?null:'Invalid model identifier'});
 if(!model)return false;
 const endpoint=await vscode.window.showInputBox({title:'Model API endpoint',value:preset.endpoint,prompt:'HTTPS endpoint or local loopback URL'});
 if(endpoint===undefined)return false;
 const key=await vscode.window.showInputBox({title:'Model API Key',prompt:'Stored only in ~/.config/voder/model-provider.json with mode 0600 on the extension host',password:true,ignoreFocusOut:true});
 if(!key)return false;
 const dest=path.join(os.homedir(),'.config/voder/model-provider.json');
 let exists=false;try{await fs.lstat(dest);exists=true;}catch(e){if(e.code!=='ENOENT')throw e;}
 if(exists){const choice=await vscode.window.showWarningMessage('Replace your existing private Voder model configuration on this extension host?',{modal:true},'Replace');if(choice!=='Replace')return false;}
 const data={VODER_MODEL_ID:model,VODER_MODEL_API_KEY:key,VODER_MODEL_VENDOR:preset.vendor,
  VODER_MODEL_PROTOCOL:preset.api,...(endpoint?{VODER_MODEL_ENDPOINT:endpoint}:{})};
 await saveModelConfig(data);
 await vscode.window.showInformationMessage('Voder model configuration saved privately on this extension host. Native VGR also requires user-scoped Modal credentials and an approved gateway profile.');
 return true;
}
module.exports={saveModelConfig,configureModel};
