/** Deterministic, evidence-priced next-action policy. An advisor, never a second Agent Loop. */
import {createHash} from 'node:crypto';
const sha = value => createHash('sha256').update(value).digest('hex');
const possible = Object.freeze({
  INSPECT:{information:5,cost:1,risk:0},
  DIAGNOSE:{information:7,cost:1,risk:0},
  PATCH:{information:6,cost:4,risk:3},
  VERIFY:{information:9,cost:6,risk:1},
  STOP:{information:0,cost:0,risk:0},
});
const unique = x => [...new Set(x)];
const compactPlan = complex => complex ? ['inspect_constraints','localize_failure','prepare_minimal_candidate','sandbox_verify','vgr_commit_or_stop'] : ['inspect_if_needed','prepare_candidate','sandbox_verify','vgr_commit_or_stop'];
/** Pure scorer: prefer reducing uncertainty per bounded cost. It does not claim global optimality. */
export function rankNextActions({stage='initial',blocked=false,budget=12,hasFailure=false}={}){
  if(!Number.isSafeInteger(budget)||budget<0||budget>64)throw new TypeError('Invalid action budget');
  const candidates=blocked||budget===0?['STOP']:
    stage==='failure'?['DIAGNOSE','INSPECT','PATCH','VERIFY']:
    stage==='verified'?['PATCH','STOP']:
    stage==='inspection'?['INSPECT','DIAGNOSE','PATCH']:
    hasFailure?['DIAGNOSE','INSPECT','PATCH']:['INSPECT','PATCH','VERIFY'];
  const scores=candidates.map(action=>{
    const base=possible[action];
    const stageFit=((stage==='failure'&&action==='DIAGNOSE')||(stage==='verified'&&action==='PATCH')||(stage==='inspection'&&action==='INSPECT'))?30:0;
    const score=action==='STOP'?0:10*base.information+stageFit-4*base.cost-6*base.risk;
    return Object.freeze({action,score,estimated_cost_units:base.cost,estimated_risk_units:base.risk});
  });
  return Object.freeze(scores.sort((a,b)=>b.score-a.score||a.action.localeCompare(b.action)));
}
export class TaskPlan {
  constructor({goal,maxActions=12,requiredFiles=[]}={}){
    if(typeof goal!=='string'||!goal.trim()||Buffer.byteLength(goal,'utf8')>8*1024*1024)throw new TypeError('Invalid task goal');
    if(!Number.isSafeInteger(maxActions)||maxActions<1||maxActions>64)throw new TypeError('Invalid action budget');
    if(!Array.isArray(requiredFiles)||requiredFiles.length>128||requiredFiles.some(x=>typeof x!=='string'))throw new TypeError('Invalid required inputs');
    this.goalDigest=sha(goal);this.maxActions=maxActions;
    this.complex=requiredFiles.length>2||/(multi[- ]file|multiple files|跨文件|多文件|架构|重构|迁移|复杂任务|multi[- ]step|repository[- ]wide)/i.test(goal);
    this.stages=Object.freeze(compactPlan(this.complex));
    this.observations=0;this.failures=0;this.revisions=0;this.blocked=false;this.lastChoice='INSPECT';this.lastReason='INITIAL';
    this.budgetConsumed=0;this.maxObservations=256;
  }
  onEvent({kind,status,progress=false,blocked=false,reason='OBSERVED'}={}){
    if(!['read','sandbox','native','exception'].includes(kind)||typeof status!=='string')throw new TypeError('Invalid plan event');
    if(this.observations>=this.maxObservations){this.blocked=true;return this.decision();}
    this.observations++;
    if(kind==='native')this.budgetConsumed++;
    if(progress)this.revisions++;
    if(blocked)this.blocked=true;
    if(kind==='exception'||status==='FAIL'||status==='BLOCKED'||status==='DENIED')this.failures++;
    this.lastReason=reason;
    this.lastChoice=this.blocked?'STOP':status==='PASS'&&kind==='sandbox'?'PATCH':kind==='native'?'STOP':
      kind==='sandbox'&&status!=='PASS'?'DIAGNOSE':kind==='exception'?'DIAGNOSE':'INSPECT';
    return this.decision();
  }
  decision(){
    const stage=this.blocked?'blocked':this.lastChoice==='DIAGNOSE'?'failure':this.lastChoice==='PATCH'?'verified':
       this.lastChoice==='INSPECT'?'inspection':'initial';
    const ranked=rankNextActions({stage,blocked:this.blocked,budget:Math.max(0,this.maxActions-this.budgetConsumed),hasFailure:this.failures>0});
    return Object.freeze({recommended:ranked[0]?.action??'STOP',ranking:ranked,reason:this.lastReason,blocked:this.blocked});
  }
  brief(){return Object.freeze({mode:this.complex?'ON_DEMAND_PLAN':'DIRECT',stages:this.stages,
    goal_digest:this.goalDigest,recommended:this.decision().recommended});}
  snapshot(){return Object.freeze({...this.brief(),observations:this.observations,failures:this.failures,
    evidence_revisions:this.revisions,native_actions:this.budgetConsumed,blocked:this.blocked});}
}
