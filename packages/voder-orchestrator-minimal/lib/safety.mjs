import {createHash, randomBytes} from 'node:crypto';
import {lstat,readFile,mkdir,open,rename,unlink,realpath} from 'node:fs/promises';
import {dirname,isAbsolute,join,resolve,sep} from 'node:path';
export const sha=b=>createHash('sha256').update(b).digest('hex');
export function insist(ok,code){if(!ok)throw new Error(code);}
export function relPath(s){insist(typeof s==='string'&&s.length>0&&s.length<=1024&&!s.startsWith('/')&&!s.includes('\\')&&!/[\x00-\x1f\x7f]/.test(s)&&s.split('/').every(p=>p!==''&&p!=='.'&&p!=='..'),'UNSAFE_RELATIVE_PATH');return s;}
export function identifier(s){insist(typeof s==='string'&&/^[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}$/.test(s),'INVALID_ID');return s;}
export async function strictDir(p){insist(isAbsolute(p),'ABSOLUTE_DIR_REQUIRED');const st=await lstat(p);insist(st.isDirectory()&&!st.isSymbolicLink(),'UNSAFE_DIRECTORY');return realpath(p);}
export async function regular(p,max=1024*1024){const st=await lstat(p);insist(st.isFile()&&!st.isSymbolicLink()&&st.nlink===1&&st.size<=max,'UNSAFE_FILE');return st;}
export async function safeWrite(p,body){const tmp=join(dirname(p),`.voder-${randomBytes(12).toString('hex')}.tmp`);let f;try{f=await open(tmp,'wx',0o600);await f.writeFile(body);await f.sync();await f.close();f=null;await rename(tmp,p);}finally{if(f)await f.close().catch(()=>{});await unlink(tmp).catch(()=>{});}}
export async function privateMkdir(p){await mkdir(p,{recursive:false,mode:0o700});await strictDir(p);}
export function isWithin(parent,child){const p=resolve(parent)+sep;return resolve(child).startsWith(p);}
export function boundedText(s,n=16_384){insist(typeof s==='string'&&Buffer.byteLength(s,'utf8')<=n,'TEXT_BUDGET_EXCEEDED');return s;}
export async function readJson(p,max=1024*1024){await regular(p,max);return JSON.parse(await readFile(p,'utf8'));}
export const redactError=e=>String(e?.message||e||'UNKNOWN').replace(/(?:sk-[a-zA-Z0-9_-]{8,}|[a-f0-9]{40,})/g,'[REDACTED]').slice(0,300);
