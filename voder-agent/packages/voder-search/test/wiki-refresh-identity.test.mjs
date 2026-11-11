import {test} from 'node:test';
import assert from 'node:assert/strict';
import {mkdtemp,mkdir,writeFile,utimes,stat,rm} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {IncrementalWiki} from '../src/index.mjs';

const BEFORE='Tool receipt must never authorize the VGR commit.\n';
const AFTER='Brand new conformance words with changed content.\n';
async function workspace(fn){
 const cwd=await mkdtemp(join(tmpdir(),'voder-wiki-identity-'));
 try {
  await mkdir(join(cwd,'docs'));
  await writeFile(join(cwd,'README.md'),'Unrelated safe repository information.\n');
  const path=join(cwd,'docs','policy.md');
  await writeFile(path,BEFORE);
  return await fn({cwd,path});
 } finally {await rm(cwd,{recursive:true,force:true});}
}

test('same-length in-place document rewrite with identical mtime and inode never reuses stale Wiki evidence',()=>workspace(async({cwd,path})=>{
 assert.equal(Buffer.byteLength(BEFORE),Buffer.byteLength(AFTER));
 const pinned=new Date('2020-01-02T03:04:05.000Z');
 await utimes(path,pinned,pinned);
 const wiki=new IncrementalWiki({cwd});
 const first=await wiki.search('receipt');
 assert.equal(first.status,'EVIDENCE_FOUND');
 const before=await stat(path);
 await writeFile(path,AFTER);
 await utimes(path,pinned,pinned);
 const after=await stat(path);
 assert.equal(before.size,after.size);
 assert.equal(before.ino,after.ino);
 assert.equal(before.mtimeMs,after.mtimeMs);
 const newDoc=await wiki.search('conformance');
 assert.equal(newDoc.status,'EVIDENCE_FOUND');
 assert.notEqual(newDoc.results[0].content_hash,first.results[0].content_hash);
 assert.equal((await wiki.search('receipt')).status,'NO_EVIDENCE');
}));

test('same metadata rewrite in the reverse direction restores exact prior evidence and never returns stale new evidence',()=>workspace(async({cwd,path})=>{
 const pinned=new Date('2020-01-02T03:04:05.000Z');
 const wiki=new IncrementalWiki({cwd});
 await utimes(path,pinned,pinned);
 const first=await wiki.search('receipt');
 await writeFile(path,AFTER);await utimes(path,pinned,pinned);
 assert.equal((await wiki.search('conformance')).status,'EVIDENCE_FOUND');
 await writeFile(path,BEFORE);await utimes(path,pinned,pinned);
 const reverted=await wiki.search('receipt');
 assert.equal(reverted.status,'EVIDENCE_FOUND');
 assert.equal(reverted.results[0].content_hash,first.results[0].content_hash);
 assert.equal((await wiki.search('conformance')).status,'NO_EVIDENCE');
}));
