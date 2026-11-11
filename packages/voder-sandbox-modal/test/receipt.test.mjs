import {test} from 'node:test';
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {receiptMetadata} from '../src/contract.ts';
test('receipt metadata never duplicates raw streams or secrets in JSON',()=>{
  const stdout='\\"\n'.repeat(1_000_000);
  const stderr='bad\\"\n'.repeat(1000);
  const hash=x=>createHash('sha256').update(x).digest('hex');
  const receipt={protocol:'voder-sandbox-receipt-v1',provider:'modal',runId:'r',actionId:'a',sandboxId:'s',image:'img',inputDigest:'digest',files:1,bytes:1,exitCode:0,timeout:false,outputLimited:false,stdoutSha256:hash(stdout),stderrSha256:hash(stderr),stdout,stderr,startedAt:'a',endedAt:'b',status:'EXECUTED'};
  const metadata=receiptMetadata(receipt);
  const encoded=JSON.stringify(metadata);
  assert.equal('stdout' in metadata,false);
  assert.equal('stderr' in metadata,false);
  assert.equal(metadata.stdoutSha256,hash(stdout));
  assert.equal(metadata.stderrSha256,hash(stderr));
  assert.ok(encoded.length<4096);
});
