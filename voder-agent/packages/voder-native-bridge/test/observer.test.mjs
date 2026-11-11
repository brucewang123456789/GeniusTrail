import { test } from 'node:test';
import assert from 'node:assert/strict';
import { NativeVgrObserver,NativeProtocolError } from '../src/index.mjs';
const valid={binary:'/tmp/voderd',stateDir:'/tmp/voder-state'};
test('invalid constructor path fails closed',()=>{assert.throws(()=>new NativeVgrObserver({binary:'./voderd',stateDir:'/tmp/state'}),{code:'INVALID_PATH'});});
test('no native authority or execute method exists',()=>{const o=new NativeVgrObserver(valid);assert.equal(o.executeAuthorized,undefined);assert.equal(o.commit,undefined);assert.equal(o.prepare,undefined);assert.equal(o.dispatch,undefined);o.close();});
test('side effects rejected before transport, malformed session denied',async()=>{const o=new NativeVgrObserver(valid);await assert.rejects(()=>o._request('native.commit'),{code:'METHOD_DENIED'});await assert.rejects(()=>o._request('tool.execute'),{code:'METHOD_DENIED'});await assert.rejects(()=>o.resume('not a valid id'),{code:'LIFECYCLE'});o.close();});
test('strict frame kind/session validation',()=>{const o=new NativeVgrObserver(valid);o.child={kill(){}};o.pending={id:'id',kind:'state.get',resolve(){throw Error('should not pass')},reject(e){assert.equal(e.code,'WIRE_MISMATCH')}};o._receive(Buffer.from(JSON.stringify({protocol_version:1,frame_type:'response',request_id:'wrong',kind:'state.get',payload:{}})+'\n'));o.close();});

test('timeout rejects pending call without replay',async()=>{
 const o=new NativeVgrObserver({...valid,timeoutMs:100});
 let sends=0;o.child={stdin:{write(_bytes,cb){sends++;cb?.();}},kill(){}};
 await assert.rejects(()=>o._request('initialize',{minimum_version:1,maximum_version:1,client_name:'probe',requested_features:[]}),{code:'TIMEOUT'});
 assert.equal(sends,1);assert.equal(o.fault.code,'TIMEOUT');o.close();
});
