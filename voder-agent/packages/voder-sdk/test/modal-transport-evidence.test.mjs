import {test} from 'node:test';
import assert from 'node:assert/strict';
import {mkdtemp,mkdir,writeFile,readFile,readdir,rm,stat} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {spawnSync} from 'node:child_process';
import {createHash} from 'node:crypto';
import {HostCandidateVerifier} from '../src/candidate-verifier.mjs';
const hash=x=>createHash('sha256').update(x).digest('hex');
const run=(cwd,...args)=>{const r=spawnSync('git',args,{cwd,encoding:'utf8',env:{...process.env,GIT_AUTHOR_NAME:'Voder Test',GIT_AUTHOR_EMAIL:'voder-test@example.invalid',GIT_COMMITTER_NAME:'Voder Test',GIT_COMMITTER_EMAIL:'voder-test@example.invalid'}});assert.equal(r.status,0,r.stderr)};
test('failed transport captures private redacted root error; no fabricated sandbox receipt',async()=>{
 const root=await mkdtemp(join(tmpdir(),'voder-evidence-'));
 const prevSecret=process.env.MODAL_TOKEN_SECRET;
 try{
  const repo=join(root,'repo'),store=join(root,'store');await mkdir(join(repo,'src'),{recursive:true});
  await writeFile(join(repo,'src/a.txt'),'OLD\n');run(repo,'init','-q');run(repo,'add','.');run(repo,'commit','-qm','baseline');
  const program='/bin/cat', args=['src/a.txt'];
  const policy=join(root,'policy.json');
  await writeFile(policy,JSON.stringify({schema:'voder.gateway.policy.v1',repository_path:repo,target_path:'src/a.txt',validator_program:program,validator_args:args,validator_sha256:hash(`${program}\0v1\0[${args.map(x=>JSON.stringify(x)).join(', ')}]`)}));
  const fakeCli=join(root,'fake-cli.mjs');
  await writeFile(fakeCli,`process.stderr.write('VODER_SANDBOX_BLOCKED=MODAL_PHASE=REMOTE_INPUT_VERIFY; Modal Sandbox is shutting down; secret='+process.env.MODAL_TOKEN_SECRET+'\\n');throw new Error('fake transport failure');\n`);
  process.env.MODAL_TOKEN_SECRET='voder-test-secret-material';
  const verifier=new HostCandidateVerifier({cwd:repo,nativeGateway:{policyFile:policy,targetPath:'src/a.txt',attemptStore:store,nodeBinary:process.execPath,modalCli:fakeCli},timeoutMs:120000});
  await assert.rejects(verifier.verifyCandidate({relative_path:'src/a.txt',old_sha256:hash('OLD\n'),new_content:'NEW\n'}),{code:'VERIFIER_INFRASTRUCTURE'});
  const dirs=await readdir(join(store,'candidate-checks'));assert.equal(dirs.length,1);
  const folder=join(store,'candidate-checks',dirs[0]);const file=join(folder,'transport-diagnostics.private.json');
  const report=JSON.parse(await readFile(file));
  assert.ok(report.child_exit_code!==0);
  assert.match(report.transport_stderr_tail,/MODAL_PHASE=REMOTE_INPUT_VERIFY/);
  assert.ok(!report.transport_stderr_tail.includes('voder-test-secret-material'));
  assert.match(report.transport_stderr_tail,/\[REDACTED\]/);
  assert.equal(report.prepare_budget_ms,720000);
  assert.equal((await stat(file)).mode&0o777,0o600);
  assert.ok(!(await readdir(folder)).includes('receipt.json'));
 }finally{
  if(prevSecret===undefined)delete process.env.MODAL_TOKEN_SECRET;else process.env.MODAL_TOKEN_SECRET=prevSecret;
  await rm(root,{recursive:true,force:true});
 }
});
