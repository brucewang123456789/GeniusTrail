/** Native OpenAI Responses adapter for models which do not implement Chat Completions.
 * No automatic retries of billable requests. VGR owns all tool execution authority.
 */
import {
  ProviderError, validateRequest, normalizeUsage,
  type GenerationRequest, type GenerationResult, type StreamEvent,
  type ModelProvider, type ProviderCapabilities, type ProposedToolCall,
} from '../../voder-providers/dist/index.js';

type Obj = Record<string,unknown>;
function obj(value: unknown): Obj {
  if (!value || typeof value !== 'object' || Array.isArray(value)) throw new ProviderError('MALFORMED_RESPONSE','expected JSON object');
  return value as Obj;
}
export class OpenAIResponsesProvider implements ModelProvider {
  readonly id: string;
  readonly #endpoint: URL;
  readonly #key: string;
  readonly #timeout: number;
  constructor(id: string, endpoint: string, key: string, timeoutMs: number) {
    if (!/^cloud_[a-z0-9_-]{1,40}$/u.test(id)) throw new TypeError('invalid provider id');
    const url = new URL(endpoint);
    if (url.protocol !== 'https:' || url.username || url.password || url.search || url.hash) throw new TypeError('Responses endpoint must be clean HTTPS');
    if (!key || /[\r\n]/u.test(key)) throw new TypeError('invalid API key');
    if (!Number.isSafeInteger(timeoutMs) || timeoutMs < 1000 || timeoutMs > 180000) throw new TypeError('invalid timeout');
    this.id = id; this.#endpoint=url; this.#key=key; this.#timeout=timeoutMs;
  }
  capabilities(): ProviderCapabilities {return {streaming:false,toolCalls:true,usageReporting:true};}
  cancel(controller: AbortController): void {controller.abort();}
  async generate(req: GenerationRequest): Promise<GenerationResult> {
    validateRequest(req);
    const url=new URL(`${this.#endpoint.pathname.replace(/\/+$/u,'')}/responses`,this.#endpoint.origin);
    const body={model:req.model,store:false,max_output_tokens:req.maxOutputTokens,
      input:req.messages.map(m=>m.role==='tool' ? {type:'function_call_output',call_id:m.toolCallId,output:m.content} : {role:m.role,content:m.content}),
      ...(req.tools?.length ? {tools:req.tools.map(t=>({type:'function',name:t.name,description:t.description,parameters:t.parameters,strict:false}))} : {}),
      ...(req.temperature!==undefined?{temperature:req.temperature}:{}),
    };
    const timeout=AbortSignal.timeout(this.#timeout);
    const signal=req.signal ? AbortSignal.any([timeout,req.signal]) : timeout;
    let response: Response;
    try{response=await fetch(url,{method:'POST',headers:{'Content-Type':'application/json','Authorization':`Bearer ${this.#key}`},body:JSON.stringify(body),signal});}
    catch(e){throw new ProviderError('UPSTREAM_UNAVAILABLE',`model transport failed (${e instanceof Error?e.name:'unknown'})`,true);}
    if (!response.ok) {
      await response.body?.cancel().catch(()=>{});
      throw new ProviderError('UPSTREAM_UNAVAILABLE',`model API HTTP ${response.status}`,response.status===429 || response.status>=500);
    }
    if (!response.body) throw new ProviderError('MALFORMED_RESPONSE','empty model body');
    const reader=response.body.getReader();
    const chunks:Uint8Array[]=[];let size=0;
    try{
      while(true){
        const {value,done}=await reader.read();
        if(done) break;
        if(value){size+=value.byteLength;if(size>512*1024) throw new ProviderError('RESPONSE_TOO_LARGE','model response exceeds 512 KiB'); chunks.push(value);}
      }
    }finally{await reader.cancel().catch(()=>{});reader.releaseLock();}
    const bytes=new Uint8Array(size);let at=0;for(const c of chunks){bytes.set(c,at);at+=c.length;}
    let parsed:unknown;
    try{parsed=JSON.parse(new TextDecoder().decode(bytes));}catch{throw new ProviderError('MALFORMED_RESPONSE','invalid JSON response');}
    const data=obj(parsed);
    if(data['status']!=='completed') throw new ProviderError('MALFORMED_RESPONSE','model response did not complete');
    if(typeof data['model']!=='string'||!data['model']) throw new ProviderError('MALFORMED_RESPONSE','missing actual model');
    if(!Array.isArray(data['output'])||data['output'].length>64) throw new ProviderError('MALFORMED_RESPONSE','invalid output');
    let text='';const tools:ProposedToolCall[]=[];
    for(const raw of data['output']){
      const item=obj(raw);
      if(item['type']==='message'){
        if(!Array.isArray(item['content'])) throw new ProviderError('MALFORMED_RESPONSE','invalid message output');
        for(const c of item['content']){
          const piece=obj(c);
          if(piece['type']==='output_text') {
            if(typeof piece['text']!=='string') throw new ProviderError('MALFORMED_RESPONSE','invalid output text');
            text+=piece['text']; if(new TextEncoder().encode(text).byteLength>256*1024) throw new ProviderError('RESPONSE_TOO_LARGE','text too large');
          }
        }
      } else if(item['type']==='function_call'){
        const id=item['call_id'],name=item['name'],args=item['arguments'];
        if(typeof id!=='string'||!id||typeof name!=='string'||!name||typeof args!=='string'||args.length>128*1024) throw new ProviderError('MALFORMED_RESPONSE','invalid function call');
        try{obj(JSON.parse(args));}catch{throw new ProviderError('MALFORMED_RESPONSE','invalid JSON tool arguments');}
        tools.push({id,name,argumentsJson:args});
      } else if(item['type']==='reasoning') {/* non-executable internal reasoning */}
      else throw new ProviderError('UNSUPPORTED','unrecognized Responses output type');
    }
    if(!text && tools.length===0) throw new ProviderError('MALFORMED_RESPONSE','empty model proposal');
    const usage=normalizeUsage(data['usage']);
    return {providerId:this.id,model:data['model'],text,toolCalls:tools,finishReason:tools.length?'tool_calls':'stop',...(usage?{usage}:{})};
  }
  async *stream(_req: GenerationRequest): AsyncIterable<StreamEvent> {
    throw new ProviderError('UNSUPPORTED','native Responses streaming not implemented');
  }
}
