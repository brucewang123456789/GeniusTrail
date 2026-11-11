import {test} from 'node:test';
import assert from 'node:assert/strict';
import {resolvePinned} from '../src/keyless-web.mjs';

const v4={address:'8.8.8.8',family:4};
const v6={address:'2606:4700:4700::1111',family:6};
function examine(records,options={all:true},err=null){
 let calls=0,passed;
 const lookup=(host,opts,cb)=>{
  assert.equal(host,'example.com');assert.deepEqual(opts,{all:true});
  cb(err,records);
 };
 resolvePinned('example.com',options,(...args)=>{calls++;passed=args;},lookup);
 assert.equal(calls,1,'callback must complete exactly once');
 return passed;
}
test('Node all:true gets an array containing only one pinned public IPv4 address',()=>{
 assert.deepEqual(examine([v4],{all:true}),[null,[v4]]);
});
test('Node all:true gets an array for IPv6-only responses',()=>{
 assert.deepEqual(examine([v6],{all:true}),[null,[v6]]);
});
test('Node all:true chooses one pinned IPv4 from safe dual-stack answers',()=>{
 assert.deepEqual(examine([v6,v4],{all:true}),[null,[v4]]);
});
test('legacy Node lookup all:false still receives positional address and family',()=>{
 assert.deepEqual(examine([v4],{all:false}),[null,v4.address,4]);
 assert.deepEqual(examine([v6],{},null),[null,v6.address,6]);
});
test('Node family selection cannot silently substitute another DNS family',()=>{
 assert.deepEqual(examine([v6,v4],{all:true,family:6}),[null,[v6]]);
 assert.deepEqual(examine([v6,v4],{all:false,family:4}),[null,v4.address,4]);
 assert.equal(examine([v4],{all:true,family:6})[0].code,'DNS_PUBLIC_ADDRESS_REQUIRED');
});
test('mixed private/public DNS replies are rejected before any network connection',()=>{
 for(const records of [[v4,{address:'127.0.0.1',family:4}],
                       [{address:'169.254.169.254',family:4},v4],
                       [v6,{address:'::ffff:127.0.0.1',family:6}]]){
  assert.equal(examine(records)[0].code,'DNS_PUBLIC_ADDRESS_REQUIRED');
 }
});
test('DNS errors, empty answers, malformed records and invalid family fail closed',()=>{
 assert.equal(examine([],{},new Error('resolver failed'))[0].code,'WEB_DNS_FAILED');
 for(const records of [[],null,[{address:'8.8.8.8',family:0}],[{address:'junk',family:4}]]){
  assert.equal(examine(records)[0].code,'DNS_PUBLIC_ADDRESS_REQUIRED');
 }
 assert.equal(examine([v4],{all:true,family:7})[0].code,'DNS_PUBLIC_ADDRESS_REQUIRED');
});
test('DNS pinning callback cannot accidentally return a raw string for all:true',()=>{
 for(const records of [[v4], [v6,v4]]){
  const [err,answer,extra]=examine(records,{all:true});
  assert.equal(err,null);assert.equal(extra,undefined);
  assert.ok(Array.isArray(answer));assert.equal(answer.length,1);
  assert.match(answer[0].address,/^[0-9a-f:.]+$/i);
 }
});
