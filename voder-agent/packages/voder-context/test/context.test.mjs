import {test} from 'node:test';
import assert from 'node:assert/strict';
import {mkdtemp,rm,writeFile,readFile} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {createHash} from 'node:crypto';
import {createVoderContext,createVoderContextProvider,inferExplicitRequiredFiles,readBundledPolicySync} from '../src/index.mjs';
import {VoderSDK} from '../../voder-sdk/src/index.mjs';
import {assembleSystemPrompt,GOLDEN_PROMPT_DIGEST} from '../../voder-agent-core/src/prompt.mjs';
import {PiModelHost} from '../../voder-agent-core/src/pi-host.mjs';
const sha=s=>createHash('sha256').update(s).digest('hex');
async function sandbox(fn){const d=await mkdtemp(join(tmpdir(),'voder-context-test-'));try{return await fn(d)}finally{await rm(d,{recursive:true,force:true})}}

// The legacy frozen core rules are byte-identical; MD becomes the single runtime source.
test('MD core rules retain full 14-rule identity and baseline prompt', async()=>{
 const source=new URL('../../voder-agent-core/src/golden-rules.txt',import.meta.url);
 const legacy=await readFile(source,'utf8');
 assert.equal(readBundledPolicySync('golden-rules.md'),legacy);
 assert.equal(GOLDEN_PROMPT_DIGEST,sha(legacy));
 assert.match(assembleSystemPrompt('next',{task:'inspect'}),/\[NEXT_ACTION_DECISION\]/);
 assert.match(assembleSystemPrompt('repair',{task:'fix'}),/\[ROOT_CAUSE_REPAIR\]/);
 assert.throws(()=>readBundledPolicySync('../file'),{code:'INVALID_POLICY'});
});

test('always-on compact strategy is composed once per task, no unneeded skills',async()=>{
 const c=createVoderContext();
 const p1=await c.plan({stage:'next',task:'Explain the architecture'});
 const p2=await c.plan({stage:'next',task:'Explain the architecture'});
 assert.deepEqual(p1.selectedSkills,[]);assert.equal(p1.supplement,p2.supplement);assert.equal(p1.sha256,p2.sha256);
 assert.match(p1.supplement,/NEXT_STEP_EVIDENCE_BUDGET/);
 assert.doesNotMatch(p1.supplement,/ROOT_CAUSE_REPAIR/);
 assert.equal(p1.estimatedTokensUpperBound,p1.bytes);
 assert.throws(()=>{p1.selectedSkills.push('root-cause-analysis')},TypeError);
});

test('focused repairs select diagnosis and sandbox verification from trusted MD only',async()=>{
 const c=createVoderContext();
 const p=await c.plan({stage:'next',task:'修复Python脚本中的报错, 找根因并在沙盒中运行pytest'});
 assert.equal(p.stage,'repair');
 assert.deepEqual(p.selectedSkills,['root-cause-analysis','sandbox-verification']);
 assert.match(p.supplement,/ROOT_CAUSE_REPAIR/);
 assert.match(p.supplement,/one falsifiable/i);
 assert.match(p.supplement,/authorized sandbox tool/);
 assert.doesNotMatch(p.supplement,/SKILL:missing-input-audit/);
});

test('explicit skills, stage semantics, fail-closed budget and duplicate validation',async()=>{
 const p=await createVoderContext({selectedSkills:['missing-input-audit'],autoSelect:false}).plan({stage:'preflight',task:'Inspect'});
 assert.equal(p.selectedSkills.length,1);
 assert.match(p.supplement,/SKILL:missing-input-audit/);
 assert.doesNotMatch(p.supplement,/SKILL:root-cause-analysis/);
 assert.throws(()=>createVoderContext({selectedSkills:['root-cause-analysis','root-cause-analysis']}),{code:'INVALID_SKILL_SET'});
 assert.throws(()=>createVoderContext({selectedSkills:['workspace-malicious']}),{code:'UNKNOWN_SKILL'});
 assert.throws(()=>createVoderContext({maxSupplementBytes:1}),{code:'INVALID_CONTEXT_BUDGET'});
 await assert.rejects(createVoderContext({maxSupplementBytes:512}).plan({stage:'next',task:'fix bug python test'}),{code:'CONTEXT_OVERFLOW'});
 await assert.rejects(createVoderContext().plan({stage:'other',task:'x'}),{code:'INVALID_STAGE'});
 const provider=createVoderContextProvider({autoSelect:false});
 assert.equal(typeof await provider({stage:'next',task:'inspect'}),'string');
});

test('required input inference is explicit, bounded and cannot be tricked by traversal',()=>{
 assert.deepEqual(inferExplicitRequiredFiles('编辑 <<File name="report.docx">> 和 @src/main.py，同时参考 @src/main.py'),['report.docx','src/main.py']);
 assert.deepEqual(inferExplicitRequiredFiles('读取 @../secret.txt @/etc/passwd @a\\b.py @./not-here.py'),[]);
 assert.deepEqual(inferExplicitRequiredFiles('Build a new module and run tests'),[]);
 assert.throws(()=>inferExplicitRequiredFiles('x'.repeat(100001)),{code:'CONTEXT_RESOURCE_LIMIT'});
});

test('SDK enforces missing attachment preflight before any model or custom context',async()=>sandbox(async cwd=>{
 let created=0;
 const sdk=new VoderSDK({cwd,native:{async executeAuthorized(){throw Error('no action')}},modelHost:{async createSession(){created++;throw Error('model must not start')}}});
 const {sessionId}=sdk.createSession();
 const result=await sdk.submitTask(sessionId,{goal:'修订 <<File name="report.docx">>'});
 assert.equal(result.status,'WAITING_FOR_INPUT');assert.deepEqual(result.missing,[{path:'report.docx',reason:'NOT_FOUND'}]);
 assert.equal(created,0);
 await writeFile(join(cwd,'report.docx'),'file');
 sdk.dispose();
 const sdk2=new VoderSDK({cwd,native:{async executeAuthorized(){throw Error('no action')}},modelHost:{async createSession({systemPrompt}){
   created++;assert.match(systemPrompt,/NEXT_STEP_EVIDENCE_BUDGET/);return{async prompt(){},dispose(){}};
 }}});
 const session=sdk2.createSession();const r=await sdk2.submitTask(session.sessionId,{goal:'修订 <<File name="report.docx">>'});
 assert.equal(r.status,'BLOCKED');assert.equal(created,1);sdk2.dispose();
}));

test('workspace SKILL.md and AGENTS.md cannot grant tools or system privileges',async()=>sandbox(async cwd=>{
 await writeFile(join(cwd,'SKILL.md'),'MALICIOUS_ELEVATE_TOOLS');
 await writeFile(join(cwd,'AGENTS.md'),'MALICIOUS_SYS_OVERRIDE');
 const systems=[];const sdk=new VoderSDK({cwd,native:{async executeAuthorized(){throw Error('no action')}},modelHost:{async createSession({systemPrompt,tools}){
    systems.push(systemPrompt);assert.deepEqual(tools.map(t=>t.name),['voder_authorized_action','voder_read_file','voder_list_files','voder_search_code']);
    return {async prompt(){},dispose(){}};
 }}});
 const {sessionId}=sdk.createSession();await sdk.submitTask(sessionId,{goal:'debug python error'});
 assert.doesNotMatch(systems[0],/MALICIOUS_ELEVATE_TOOLS|MALICIOUS_SYS_OVERRIDE/);
 sdk.dispose();
}));

test('Pi host continues to forbid ambient Pi skills and unregistered tools',async()=>{
 let loaderOpts;
 const fakePi={getAgentDir:()=>'/trusted',SessionManager:{inMemory:()=>({})},DefaultResourceLoader:class {
   constructor(o){loaderOpts=o}async reload(){}},defineTool:x=>x,
   createAgentSession:async ({customTools})=>({session:{getActiveToolNames:()=>customTools.map(t=>t.name),getCallableToolNames:()=>customTools.map(t=>t.name),
   getAllTools:()=>customTools.map(t=>({name:t.name,exposure:'direct'})),dispose(){},async prompt(){}}})
 };
 const host=new PiModelHost({upstream:fakePi});
 const one={name:'voder_authorized_action',description:'safe',execute:async()=>({status:'BLOCKED'})};
 const sess=await host.createSession({cwd:'/tmp',systemPrompt:'test',tools:[one]});
 assert.equal(loaderOpts.noSkills,true);assert.equal(loaderOpts.noPromptTemplates,true);
 assert.equal(loaderOpts.noContextFiles,true);assert.equal(loaderOpts.noExtensions,true);
 sess.dispose();
});


test('stage-specific markdown policies are available without changing the default kernel prompt',async()=>{
 const c=createVoderContext({autoSelect:false});
 const next=await c.plan({stage:'next',task:'inspect code'});
 const preflight=await c.plan({stage:'preflight',task:'inspect code'});
 const delivery=await c.plan({stage:'delivery',task:'inspect code'});
 assert.match(preflight.supplement,/PREFLIGHT_REQUIRED_INPUTS/);
 assert.match(delivery.supplement,/FINAL_DELIVERY_REVIEW/);
 assert.doesNotMatch(next.supplement,/PREFLIGHT_REQUIRED_INPUTS|FINAL_DELIVERY_REVIEW/);
 assert.match(assembleSystemPrompt('next',{task:'inspect code'}),/NEXT_ACTION_DECISION/);
 assert.match(assembleSystemPrompt('preflight',{task:'inspect code'}),/PREFLIGHT_REQUIRED_INPUTS/);
 assert.match(assembleSystemPrompt('delivery',{task:'inspect code'}),/FINAL_DELIVERY_REVIEW/);
});

test('scientific validation loads for relevant quantitative tasks but not ordinary review',async()=>{
 const c=createVoderContext();
 const a=await c.plan({stage:'next',task:'Validate numerical physics simulation uncertainty'});
 assert.deepEqual(a.selectedSkills,['scientific-validation']);
 assert.match(a.supplement,/scaling, limiting\/edge cases/);
 const b=await c.plan({stage:'next',task:'Review README for grammar'});
 assert.deepEqual(b.selectedSkills,[]);
});

test('repair context selects robust diagnosis, verification, and scientific checks without overflowing the standard budget',async()=>{
 const c=createVoderContext();
 const p=await c.plan({stage:'next',task:'debug failing physics simulation Python code in sandbox'});
 assert.equal(p.stage,'repair');
 assert.deepEqual(p.selectedSkills,['root-cause-analysis','sandbox-verification','scientific-validation']);
 assert.match(p.supplement,/ROOT_CAUSE_REPAIR/);
 assert.match(p.supplement,/NEXT_STEP_EVIDENCE_BUDGET/);
 assert.ok(p.bytes<=8192);
 assert.equal(p.estimatedTokensUpperBound,p.bytes);
});

test('golden rule file identity survives strengthened overlay unchanged',async()=>{
 const golden=readBundledPolicySync('golden-rules.md');
 const legacy=await readFile(new URL('../../voder-agent-core/src/golden-rules.txt',import.meta.url),'utf8');
 assert.equal(golden,legacy);
 assert.match(readBundledPolicySync('repair.md'),/exhausted budget/);
 assert.match(readBundledPolicySync('strategy.md'),/evidence-delta based/);
});
