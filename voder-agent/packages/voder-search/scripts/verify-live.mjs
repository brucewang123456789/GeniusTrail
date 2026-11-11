/** Real tool-path check: at most two search requests and one direct page fetch.
 * No model calls. Does not infer laureates or produce a factual verification receipt.
 */
import {EvidenceSearchService} from '../src/index.mjs';
import {writeFileSync} from 'node:fs';
const args=process.argv.slice(2);const flag=name=>{const i=args.indexOf(name);return i<0?undefined:args[i+1]};
const override=flag('--query'),requiredSource=flag('--source');
const goal='请搜索网页，核实2026年诺贝尔物理学奖得主是谁，必须以诺贝尔奖官方网站为依据，并给出获奖理由、具体贡献及来源链接。';
const service=new EvidenceSearchService({cwd:process.cwd(),allowWeb:true,webMaxQueries:3,webCacheTtlMs:0,negativeCacheTtlMs:0});
const task=service.newTask(override?'web search validation: '+override:goal);
const report={model_calls:0,prompt:override??goal,status:'STARTED',started_at:new Date().toISOString()};
try{
 const discovery=await service.query(task,{mode:'web',query:override??'site:nobelprize.org/prizes/physics/2026 2026 Nobel Prize Physics laureates'});
 report.discovery=discovery;
 const hit=discovery.results?.find(r=>{if(requiredSource)return r.url===requiredSource;if(override)return true;const u=new URL(r.url);return (u.hostname==='nobelprize.org'||u.hostname==='www.nobelprize.org')&&u.pathname.startsWith('/prizes/physics/2026/');});
 if(!hit)throw Object.assign(new Error('WEB_SEARCH_NO_SCOPED_RESULT'),{code:'WEB_SEARCH_NO_SCOPED_RESULT'});
 report.page=await service.query(task,{mode:'fetch',query:hit.url,focus:'laureates Nobel Prize Physics 2026'});
 if(report.page.status!=='EVIDENCE_FOUND')throw Object.assign(new Error('WEB_PAGE_NOT_FETCHED'),{code:'WEB_PAGE_NOT_FETCHED'});
 report.status='SEARCH_AND_LOCAL_FETCH_PASS';
}catch(e){report.discovery_attempts=service.provider?.web?.lastAttempts??[];if(e?.network_receipt)report.failure_network_receipt=e.network_receipt;report.status='FAILED';report.code=/^[A-Z_]+$/.test(e?.code??'')?e.code:'WEB_PROVIDER_FAILED';process.exitCode=2;}
report.search_transport_diagnostics=service.provider?.web?.lastAttempts??[];report.fetch_transport_diagnostics=service.provider?.web?.lastFetchReceipts??[];
report.finished_at=new Date().toISOString();report.budget={used:task.webUsed,search_used:task.webSearchUsed,fetch_used:task.webFetchUsed};
if(process.argv[2])writeFileSync(process.argv[2],JSON.stringify(report,null,2),{mode:0o600});
console.log('LIVE_RESULT='+report.status+(report.code?'; CODE='+report.code:'')+'; MODEL_CALLS=0; SEARCH_REQUESTS='+task.webSearchUsed+'; LOCAL_FETCH_REQUESTS='+task.webFetchUsed);
