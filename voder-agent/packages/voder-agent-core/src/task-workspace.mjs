import {createHash} from 'node:crypto';
/** Trusted, host-installed task capabilities. No model-selected executable or VGR authority. */
import {VoderError} from './contracts.mjs';
const approved=new WeakSet(),pictures=new WeakMap();
export const taskToolRequestAck=result=>pictures.get(result)?.onRequested;
export const hasTaskToolImages=result=>(pictures.get(result)?.images.length??0)>0;
export const isHostTaskTool=t=>!!t&&approved.has(t);
export function taskToolImages(result,{supported=false}={}){
 const entry=pictures.get(result);if(!entry||!supported)return [];
 entry.onExposed?.();return entry.images;
}
export class HostTaskWorkspace {
 constructor({prepare,render,review,hash,assess,browser,commitReadiness,requireVisualDecision=false}={}){
  if([prepare,render,review,hash,assess].some(x=>typeof x!=='function'))throw new VoderError('INVALID_TASK_WORKSPACE','Host task callbacks required');
  if(typeof requireVisualDecision!=='boolean'&&typeof requireVisualDecision!=='function')throw new VoderError('INVALID_TASK_WORKSPACE','Visual decision policy must be boolean or trusted callback');
  if(commitReadiness!==undefined&&typeof commitReadiness!=='function')throw new VoderError('INVALID_TASK_WORKSPACE','Trusted commit readiness callback required');
  if(browser!==undefined&&typeof browser!=='function')throw new VoderError('INVALID_TASK_WORKSPACE','Optional browser callback must be host-installed');
  Object.assign(this,{prepare,render,review,hash,assess,browser,commitReadiness,requireVisualDecision});
 }
 makeTools({stillCurrent=()=>{},beforePrepare=()=>{},observe=()=>{},afterReview=async()=>{}}={}){
  let calls=0;
  const visualDecisionMandatory=typeof this.requireVisualDecision==='function'?this.requireVisualDecision():this.requireVisualDecision;
  if(typeof visualDecisionMandatory!=='boolean')throw new VoderError('INVALID_TASK_WORKSPACE','Visual decision callback must return boolean');
  const specs=[
   ['voder_task_prepare','Prepare or extend exact source file scope INSIDE the approved workspace, during this executing task. Optional; use only when writing. Creates tracked host preimages for absent text sources, preserving existing bytes. Optional checks are exact argv arrays executed ONLY in the independent sandbox against each candidate, never on the host. Returns actual SHA256 values. No final JSON plan is required. Include all intended UTF-8 source files; PNG previews are derived by the SVG renderer, not text candidates.',{paths:{type:'array',minItems:1,maxItems:16,items:{type:'string'}},checks:{type:'array',maxItems:8,items:{type:'array',items:{type:'string'}}}},['paths'],async a=>{beforePrepare();return this.prepare(a.paths,a.checks);}],
   ['voder_task_render_svg','Render proposed self-contained SVG WITHOUT committing it. Returns exact source hash, full-size and reduced PNG images when a renderer and vision model are available. Repair significant defects, render again, then record review for the FINAL hash. No image generation service is used. Optional preview_path is a workspace-relative .png derived only after verified source delivery.',{path:{type:'string'},content:{type:'string'},preview_path:{type:'string'}},['path','content'],async a=>{
    const {result,images=[],onExposed,onRequested}=await this.render(a);pictures.set(result,{images,onExposed,onRequested});return result;
   }],
   ['voder_task_review_render','Record YOUR visual observations for the exact rendered source hash after actually viewing the returned images or browser frames. Include visual_decision ACCEPT only when the final output meets the requested visual requirements; choose REPAIR_REQUIRED for defects, or NEEDS_MORE_EVIDENCE when a specific additional view/control/phase test is necessary. For NEEDS_MORE_EVIDENCE state the missing evidence in limitations, then inspect a novel check or probe; this does not authorize commit. This is a model judgment, NEVER an independent quality certificate. Host evidence-mode commit requires exact-hash ACCEPT and independent PASS.',{source_sha256:{type:'string'},findings:{type:'string'},limitations:{type:'string'},visual_decision:{type:'string',enum:['ACCEPT','REPAIR_REQUIRED','NEEDS_MORE_EVIDENCE']}},visualDecisionMandatory?['source_sha256','findings','limitations','visual_decision']:['source_sha256','findings','limitations'],a=>this.review(a)],
   ['voder_task_content_hash','Compute actual SHA256 of exact proposed UTF-8 text. Does not write or validate it. Use for verification records, never guess hashes.',{content:{type:'string'}},['content'],a=>this.hash(a.content)]
  ];
  if(this.browser)specs.push(['voder_task_inspect_html','Execute supplied self-contained HTML in a fresh network-blocked remote browser sandbox (no cumulative execution-count limit; 3600 seconds without progress stops the task). Returns exact HTML hash, real Chromium screenshots, console/runtime evidence, optional bounded page-side checks, deterministic phase samples, and timed playback samples. Supply optional probe function paths matching your application API; no scene-specific global is assumed. It is inspection evidence, NEVER a candidate verifier PASS or a Rust commit. Call only after source scope preparation for file-creation tasks. Use review_render with the final HTML hash after seeing returned images. preview_path optionally reserves one PNG derived only after exact source delivery.',{path:{type:'string'},content:{type:'string'},probe:{type:'object',properties:{phase_function:{type:'string'},state_function:{type:'string'},controls:{type:'object',properties:{play:{type:'string'},pause:{type:'string'},restart:{type:'string'}},additionalProperties:false}},additionalProperties:false},checks:{type:'array',maxItems:8,items:{type:'string'}},playback_seconds:{type:'number',minimum:0,maximum:24},preview_path:{type:'string'}},['path','content'],async a=>{const {result,images=[],onExposed,onRequested}=await this.browser(a);pictures.set(result,{images,onExposed,onRequested});return result;}]);
  return specs.map(([name,description,properties,required,run])=>{
   const tool=Object.freeze({name,description,parameters:{type:'object',properties,required,additionalProperties:false},execute:async(args,signal)=>{
    stillCurrent();if(this.failure)throw this.failure;if(signal?.aborted)throw new VoderError('CANCELLED','Task cancelled');
    if(!args||typeof args!=='object'||Array.isArray(args)||Object.keys(args).some(k=>!Object.hasOwn(properties,k))||required.some(k=>!Object.hasOwn(args,k)))throw new VoderError('TASK_TOOL_ARGUMENTS','Unexpected task tool arguments');
    if(name==='voder_task_review_render'&&visualDecisionMandatory&&!['ACCEPT','REPAIR_REQUIRED','NEEDS_MORE_EVIDENCE'].includes(args.visual_decision))throw new VoderError('VISUAL_DECISION_REQUIRED','Review the actual frames, then explicitly select ACCEPT, REPAIR_REQUIRED, or NEEDS_MORE_EVIDENCE. Never invent acceptance; provide a concrete limitation when more evidence is needed.');
    const {content:_source,...metadata}=args;const raw=JSON.stringify(metadata);if(Buffer.byteLength(raw)>180000)throw new VoderError('TASK_TOOL_SIZE','Narrow task tool input');
    if(++calls>64&&!this.browser)throw new VoderError('TASK_CAPABILITY_BUDGET','Task capability budget exhausted; do not replay the task');
    let result;try{result=await run({...args,signal});}catch(e){if(e?.terminalTaskBlocker===true)this.failure=e;throw e;}stillCurrent();observe({tool:name,status:result?.status??'OBSERVED',observation_sha256:createHash('sha256').update(JSON.stringify({name,args,result})).digest('hex')});
    if(name==='voder_task_review_render'&&args.visual_decision==='ACCEPT'){const outcome=await afterReview(result,signal);stillCurrent();if(outcome?.status)return {...result,host_commit:outcome};}
    return result;
   }});approved.add(tool);return tool;
  });
 }
}
