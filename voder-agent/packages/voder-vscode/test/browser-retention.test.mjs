/** Offline request projection regression; no provider or remote sandbox. */
import{test}from'node:test';import assert from'node:assert/strict';
import{PiModelHost}from'../../voder-agent-core/src/pi-host.mjs';
import{HostTaskWorkspace}from'../../voder-agent-core/src/task-workspace.mjs';
const original={type:'image',mimeType:'image/png',data:'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jXioAAAAASUVORK5CYII='};
const native={name:'voder_authorized_action',description:'Offline native boundary',execute(){throw Error('No real native execution');}};
async function scenario({change=false,distinct=false}={}){
 let acknowledgements=0,round=0,wire=[],session;
 const workspace=new HostTaskWorkspace({prepare:async()=>({}),review:async()=>({}),hash:async()=>({}),assess:async()=>({}),render:async()=>({result:{}}),browser:async()=>({result:{status:'BROWSER_INSPECTED'},images:Array.from({length:8},(_,i)=>({...original,data:Buffer.from('frame-'+round+'-'+i).toString('base64')})),onRequested(){acknowledgements++;}})});
 const api={defineTool:x=>x,getAgentDir:()=>'/fixture',SessionManager:{inMemory:()=>({})},normalizeContext:x=>x,DefaultResourceLoader:class{async reload(){}},async createAgentSession(cfg){
  const names=cfg.tools;session={messages:[],model:{provider:'fixture',id:'vision',input:['text','image'],contextWindow:2000000,maxTokens:1024},systemPrompt:'Rules',getActiveToolNames:()=>names,getCallableToolNames:()=>names,getAllTools:()=>names.map(name=>({name,exposure:'direct'})),subscribe:()=>()=>{},dispose(){},agent:{streamFunction:(_m,c)=>{wire.push(structuredClone(c));return{};}},async prompt(goal,options){
   this.messages.push({role:'user',content:[{type:'text',text:goal},...options.images]});
   const send=()=>this.agent.streamFunction(this.model,{messages:[{role:'system',content:'Rules'},...this.messages]},{});
   send();
   for(round=0;round<5;round++){
    const tool=cfg.customTools.find(t=>t.name==='voder_task_inspect_html'),id='render-'+round;
    const result=await tool.execute(id,{path:distinct&&round===4?'other.html':'index.html',content:'<!DOCTYPE html><html></html>'});
    this.messages.push({role:'toolResult',toolCallId:id,toolName:tool.name,content:result.content});send();
   }
   if(change){this.messages.find(m=>m.role==='toolResult').content.find(b=>b.type==='image').data='changed';send();}
   this.messages.push({role:'assistant',stopReason:'stop',content:[{type:'text',text:'Fixture response'}]});
  }};return{session};}};
 const host=await new PiModelHost({upstream:api}).createSession({cwd:process.cwd(),systemPrompt:'Rules',tools:[native,...workspace.makeTools()]});
 try{if(change)await assert.rejects(host.prompt('Inspect',{images:[original]}),{code:'REFERENCE_IMAGE_CHANGED'});else await host.prompt('Inspect',{images:[original]});}
 finally{host.dispose();}
 return{wire,session,acknowledgements};
}
test('five browser rounds send only latest eight frames plus unchanged user attachment; transcript remains intact',async()=>{
 const r=await scenario();assert.equal(r.acknowledgements,5);
 assert.deepEqual(r.wire.map(c=>c.messages.flatMap(m=>m.content??[]).filter(b=>b.type==='image').length),[1,9,9,9,9,9]);
 assert.equal(r.session.messages.flatMap(m=>m.content??[]).filter(b=>b.type==='image').length,41);
 for(const c of r.wire)assert.deepEqual(c.messages.find(m=>m.role==='user').content[1],original);
 const latest=r.wire.at(-1);assert.equal(latest.messages.filter(m=>m.role==='toolResult'&&m.content.some(b=>b.type==='image')).length,1);
 assert.ok(latest.messages.find(m=>m.toolCallId==='render-0').content.some(b=>b.text?.includes('Superseded')));
});
test('superseded frame corruption is rejected before projection, preserving image integrity checks',async()=>{const r=await scenario({change:true});assert.equal(r.wire.length,6);});

test('a different source path keeps its own latest browser frames',async()=>{const r=await scenario({distinct:true});assert.equal(r.wire.at(-1).messages.flatMap(m=>m.content??[]).filter(b=>b.type==='image').length,17);});
