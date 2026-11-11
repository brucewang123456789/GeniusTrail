/** Operator-owned provider catalog. Secrets never enter model selection receipts or logs. */
import {open,lstat} from 'node:fs/promises';import {constants} from 'node:fs';import {join,dirname,resolve} from 'node:path';import {homedir} from 'node:os';
const own=(x,k)=>Object.hasOwn(x,k);const fail=code=>{throw Error(code);};
export const defaultCatalogPath=()=>join(homedir(),'.config/voder/additional-models.json');
export function validateCatalog(data,env=process.env){
 if(!data||data.version!==1||Object.keys(data).some(k=>!['version','models'].includes(k))||!Array.isArray(data.models))fail('INVALID_ADDITIONAL_MODEL_CATALOG');
 const seen=new Set();return data.models.map(p=>{
  const keys=['provider','id','api','baseUrl','apiKey','apiKeyEnv','contextWindow','maxTokens','reasoning','thinkingLevel','compat','input','thinkingLevelMap'];
  if(!p||typeof p!=='object'||Array.isArray(p)||Object.keys(p).some(k=>!keys.includes(k)))fail('INVALID_ADDITIONAL_MODEL');
  if(!/^voder-secondary-[a-z0-9.-]{1,48}$/.test(p.provider)||!/^[-\w./:+]{1,128}$/.test(p.id)||seen.has(p.provider+'\0'+p.id))fail('INVALID_ADDITIONAL_MODEL_ID');seen.add(p.provider+'\0'+p.id);
  if(!['openai-completions','openai-responses','anthropic-messages','google-generative-ai'].includes(p.api))fail('UNSUPPORTED_SECONDARY_PROTOCOL');
  let url;try{url=new URL(p.baseUrl);}catch{fail('INVALID_ADDITIONAL_MODEL_ENDPOINT');}
  if(!(url.protocol==='https:'||url.protocol==='http:'&&['127.0.0.1','[::1]','localhost'].includes(url.hostname))||url.username||url.password||url.search||url.hash||/%2f|%5c|%2e/i.test(url.pathname))fail('INVALID_ADDITIONAL_MODEL_ENDPOINT');
  if(!Number.isSafeInteger(p.contextWindow)||p.contextWindow<1024||p.contextWindow>1048576||!Number.isSafeInteger(p.maxTokens)||p.maxTokens<64||p.maxTokens>p.contextWindow)fail('INVALID_ADDITIONAL_MODEL_LIMITS');
  if(typeof p.reasoning!=='boolean'||!['off','low','high','max'].includes(p.thinkingLevel)||(!p.reasoning&&p.thinkingLevel!=='off'))fail('INVALID_ADDITIONAL_MODEL_REASONING');
  if(own(p,'apiKey')===own(p,'apiKeyEnv'))fail('EXACTLY_ONE_SECONDARY_CREDENTIAL_REQUIRED');
  if(p.apiKeyEnv!==undefined&&!/^[A-Z][A-Z0-9_]{0,79}$/.test(p.apiKeyEnv))fail('INVALID_SECONDARY_CREDENTIAL_REFERENCE');
  const apiKey=p.apiKeyEnv?env[p.apiKeyEnv]:p.apiKey;
  if(typeof apiKey!=='string'||apiKey.length<8||apiKey.length>4096||/[\r\n\0]/.test(apiKey))fail('SECONDARY_CREDENTIAL_MISSING_OR_INVALID');
  if(p.thinkingLevelMap!==undefined&&(typeof p.thinkingLevelMap!=='object'||p.thinkingLevelMap===null||Array.isArray(p.thinkingLevelMap)||Object.entries(p.thinkingLevelMap).some(([k,v])=>!['off','minimal','low','medium','high','xhigh','max'].includes(k)||(v!==null&&(typeof v!=='string'||!/^[-\w.]{1,32}$/.test(v))))))fail('INVALID_SECONDARY_THINKING_MAP');
  if(p.input!==undefined&&(!Array.isArray(p.input)||!p.input.includes('text')||new Set(p.input).size!==p.input.length||p.input.some(x=>!['text','image'].includes(x))))fail('INVALID_SECONDARY_MODALITIES');
  const allowedCompat=['thinkingFormat','supportsReasoningEffort','supportsDeveloperRole','zaiToolStream','supportsStore','supportsUsageInStreaming','requiresToolResultName','requiresAssistantAfterToolResult','requiresThinkingAsText','supportsStrictMode','maxTokensField'];const compat=p.compat??{};
  if(typeof compat!=='object'||Array.isArray(compat)||Object.keys(compat).some(k=>!allowedCompat.includes(k))||Object.entries(compat).some(([k,v])=>k==='thinkingFormat'?!['zai','deepseek','qwen','openrouter'].includes(v):k==='maxTokensField'?!['max_tokens','max_completion_tokens'].includes(v):typeof v!=='boolean'))fail('INVALID_SECONDARY_COMPAT');
  if(p.id.toLowerCase()==='glm-5.3-flash'&&(!p.reasoning||p.thinkingLevel==='off'||compat.thinkingFormat!=='zai'||compat.supportsReasoningEffort!==true))fail('GLM_53_FLASH_REQUIRES_ENABLED_REASONING');
  return Object.freeze({...p,baseUrl:url.href.replace(/\/$/,''),apiKey,input:Object.freeze(p.input??['text']),compat:Object.freeze({...compat})});
 });
}
export async function readAdditionalModels({file=defaultCatalogPath(),env=process.env}={}){
 // Check every parent as well as the opened descriptor; never follow config symlinks.
 const full=resolve(file);let fd;
 try{fd=await open(full,constants.O_RDONLY|constants.O_NOFOLLOW);}catch(e){if(e.code==='ENOENT')return [];throw Error('UNSAFE_SECONDARY_MODEL_CATALOG');}
 try{let parent=dirname(full);while(true){const st=await lstat(parent);if(!st.isDirectory()||st.isSymbolicLink()||![0,process.getuid()].includes(st.uid)||((st.mode&0o022)&&!(st.mode&0o1000)))fail('UNSAFE_SECONDARY_CATALOG_DIRECTORY');const next=dirname(parent);if(next===parent)break;parent=next;}const st=await fd.stat();if(!st.isFile()||st.uid!==process.getuid()||(st.mode&0o077)||st.size>65536)fail('UNSAFE_SECONDARY_MODEL_CATALOG');let data;try{data=JSON.parse(await fd.readFile('utf8'));}catch{fail('INVALID_SECONDARY_CATALOG_JSON');}return validateCatalog(data,env);}finally{await fd.close();}
}
