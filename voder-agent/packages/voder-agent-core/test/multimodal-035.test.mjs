import {test} from 'node:test';
import assert from 'node:assert/strict';
import {PiModelHost} from '../src/pi-host.mjs';
import {guardedStream} from '../src/request-policy.mjs';
import {modelEventMetadata} from '../src/model-diagnostics.mjs';
import {tmpdir} from 'node:os';
const image={type:'image',data:'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jXioAAAAASUVORK5CYII=',mimeType:'image/png'};
const native={name:'voder_authorized_action',description:'No native calls in this test',parameters:{type:'object'},execute:async()=>{throw Error('NOT_AUTHORIZED');}};
function fixture({drop=false,transform=false}={}){
 const text={provider:'fixture',id:'text',input:['text'],contextWindow:1048576,maxTokens:1024};
 const vision={provider:'voder-secondary-vision',id:'vision',input:['text','image'],contextWindow:1048576,maxTokens:1024};
 const api={calls:0,getAgentDir:()=>'/fixture',defineTool:x=>x,SessionManager:{inMemory:()=>({})},normalizeContext:x=>x,DefaultResourceLoader:class{async reload(){}},createAgentSession:async cfg=>{
  const names=cfg.customTools.map(x=>x.name);const session={model:text,systemPrompt:cfg.systemPrompt,messages:[],getActiveToolNames:()=>names,getCallableToolNames:()=>names,getAllTools:()=>names.map(name=>({name,exposure:'direct'})),dispose(){},subscribe(){return ()=>{};},abort(){},async setModel(m){this.model=m;},agent:{state:{},streamFunction:()=>{api.calls++;return {}; }},async prompt(goal,options){
   const blocks=drop?[]:options.images.map(x=>transform?{...x,data:Buffer.from('changed').toString('base64')}:x);this.messages.push({role:'user',content:[{type:'text',text:goal},...blocks]});this.agent.streamFunction(this.model,{messages:[{role:'system',content:cfg.systemPrompt},...this.messages]},{});this.messages.push({role:'assistant',stopReason:'stop',content:[{type:'text',text:'fixture answer'}]});
  }};api.session=session;return {session};}};
 return {api,vision};
}
test('image-bearing task selects the dedicated approved vision model and emits matching digest metadata',async()=>{
 const {api,vision}=fixture();const host=await new PiModelHost({upstream:api,modelProfiles:[vision]}).createSession({cwd:tmpdir(),systemPrompt:'RULES',tools:[native]});const events=[];host.subscribe(x=>events.push(x));
 try{await host.prompt('ordinary reading',{images:[image]});assert.equal(api.calls,1);assert.equal(api.session.model.provider,'voder-secondary-vision');const e=events.find(x=>x.type==='image_request');assert.equal(e.image_count,1);assert.match(e.images_sha256,/^[a-f0-9]{64}$/);assert.equal(modelEventMetadata(e).status,'ADAPTER_VALIDATED_NOT_PROVIDER_ACK');}finally{host.dispose();}
});
for(const mode of [{drop:true},{transform:true}])test('vision routing never permits dropped or modified images '+JSON.stringify(mode),async()=>{
 const {api,vision}=fixture(mode);const host=await new PiModelHost({upstream:api,modelProfiles:[vision]}).createSession({cwd:tmpdir(),systemPrompt:'RULES',tools:[native]});try{await assert.rejects(host.prompt('read',{images:[image]}),{code:'REFERENCE_IMAGE_CHANGED'});assert.equal(api.calls,0);}finally{host.dispose();}
});
test('missing approved vision model blocks before a provider request',async()=>{const {api}=fixture(),host=await new PiModelHost({upstream:api}).createSession({cwd:tmpdir(),systemPrompt:'RULES',tools:[native]});try{await assert.rejects(host.prompt('read',{images:[image]}),{code:'REFERENCE_IMAGE_UNSUPPORTED'});assert.equal(api.calls,0);}finally{host.dispose();}});
test('image base64 is preserved on the wire and excluded from text-token estimate',()=>{
 const model={provider:'f',id:'f',contextWindow:100000,maxTokens:1024},b={...image,data:'A'.repeat(500000)};let got,audit;
 const guarded=guardedStream({raw:(m,c)=>{got=c;return {};},normalizeContext:x=>x,systemPrompt:'RULES',inspect(){},allowedModels:[model],currentModel:()=>model,onAudit:x=>audit=x});
 guarded(model,{messages:[{role:'system',content:'RULES'},{role:'user',content:[b]}]},{});assert.equal(got.messages[1].content[0].data,b.data);assert.equal(audit.image_count,1);assert.equal(audit.image_token_allowance,16384);assert.ok(audit.context_estimate_tokens<20000);
});

test('same-session continuation retains earlier images without a false dropped-image error',async()=>{const {api,vision}=fixture();const host=await new PiModelHost({upstream:api,modelProfiles:[vision]}).createSession({cwd:tmpdir(),systemPrompt:'RULES',tools:[native]});try{await host.prompt('first',{images:[image]});await host.prompt('second',{images:[image]});assert.equal(api.calls,2);assert.equal(api.session.messages.filter(x=>x.role==='user').length,2);}finally{host.dispose();}});

test('actual JSON-lines RPC and SDK kernel retain 45 document images in a frame larger than 1 MiB',async()=>{
 const {createRequire}=await import('node:module'),{PassThrough}=await import('node:stream'),{mkdtemp,rm}=await import('node:fs/promises'),{join,resolve}=await import('node:path');
 const {VoderSDK}=await import('../../voder-sdk/src/index.mjs'),{serveRpc}=await import('../../voder-cli/bin/voder-cli.mjs');
 const require=createRequire(import.meta.url),{ReferenceController}=require('../../voder-vscode/src/reference-ui.cjs'),refsContract=require('../src/reference-context.cjs');
 const fixtures=process.env.VODER_DOCUMENT_FIXTURES||resolve('fixtures'),file=join(fixtures,'forty-five.pdf');
 const refs=await new ReferenceController({vscode:{window:{},workspace:{textDocuments:[]}}}).requestContext({uri:{fsPath:fixtures}},'概括全文',[{value:{uri:{scheme:'file',fsPath:file}}}]);
 const content='RPC_PRIVATE_SENTINEL'+'.'.repeat(1100000);
 const references=[...refs,{id:'00000000-3333',name:'large.txt',kind:'text',source:'attachment',content,sha256:refsContract.digest(Buffer.from(content))}];
 const cwd=await mkdtemp(join(tmpdir(),'voder-035-rpc-'));let seen=0,nativeCalls=0;
 try{
  const sdk=new VoderSDK({cwd,allowUnverifiedDraft:true,native:{async executeAuthorized(){nativeCalls++;throw Error('DENIED');}},modelHost:{async createSession(){return {referenceContextImages:true,async prompt(goal,{images}){seen=images.length;return {draft:'IMAGE_COUNT='+images.length};}};}}});
  const {sessionId}=sdk.createSession(),input=new PassThrough(),output=new PassThrough(),chunks=[];output.on('data',x=>chunks.push(x));
  const server=serveRpc({sdk,submitTask:async(id,task)=>sdk.submitTask(id,task),shutdown:async()=>{}},{input,output});
  const request=JSON.stringify({jsonrpc:'2.0',id:1,method:'task.submit',params:{sessionId,task:{goal:'Read declared document',referenceContext:references,maxActions:1}}});assert.ok(Buffer.byteLength(request)>1048576);
  input.end(request+'\n');await server;const response=JSON.parse(Buffer.concat(chunks).toString().trim());assert.equal(response.result.status,'DRAFT_UNVERIFIED');assert.equal(response.result.verified,false);assert.equal(seen,45);assert.equal(nativeCalls,0);
 }finally{await rm(cwd,{recursive:true,force:true});}
});
