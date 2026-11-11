/** Restricted first-party evidence transport. Read only; no model-controlled domains. */
import {request} from 'node:https';
import {lookup} from 'node:dns';
import {isIP,BlockList} from 'node:net';
const allowed=new Set(['docs.python.org','doc.rust-lang.org','www.typescriptlang.org','typescriptlang.org','pkg.go.dev','go.dev']);
const deny=new BlockList();
for(const [ip,bits] of [['0.0.0.0',8],['10.0.0.0',8],['100.64.0.0',10],['127.0.0.0',8],['169.254.0.0',16],['172.16.0.0',12],['192.0.0.0',24],['192.0.2.0',24],['192.168.0.0',16],['198.18.0.0',15],['198.51.100.0',24],['203.0.113.0',24],['224.0.0.0',4],['240.0.0.0',4]])deny.addSubnet(ip,bits,'ipv4');
for(const [ip,bits] of [['::',128],['::1',128],['fc00::',7],['fe80::',10],['ff00::',8],['2001:db8::',32]])deny.addSubnet(ip,bits,'ipv6');
function fail(code){let e=new Error(code);e.code=code;throw e;}
export function publicIp(address){const fam=isIP(address);return fam>0 && !(fam===6&&address.toLowerCase().includes(':ffff:'))&&!deny.check(address,fam===4?'ipv4':'ipv6');}
export function officialUrl(raw){let u;try{u=new URL(raw)}catch{fail('INVALID_OFFICIAL_URL')};if(u.protocol!=='https:'||u.port||u.username||u.password||!allowed.has(u.hostname)||!/^\/[a-zA-Z0-9%_.,+~!$&'()*;=:@/?#\-]*$/.test(u.pathname+u.search+u.hash))fail('INVALID_OFFICIAL_URL');return u;}
export function pinnedLookup(host,options,callback){
 lookup(host,{all:true},(error,addresses)=>{
  if(error||!addresses?.length||addresses.some(a=>!publicIp(a.address)))return callback(Object.assign(new Error('NONPUBLIC_DNS_RESPONSE'),{code:'NONPUBLIC_DNS_RESPONSE'}));
  const selected=addresses.find(x=>x.family===4)??addresses[0];
  if(options?.all)callback(null,[{address:selected.address,family:selected.family}]);
  else callback(null,selected.address,selected.family);
 });
}
export function officialGet(url,{signal,limit=3*1024*1024,timeout=12000}={}){
 const u=officialUrl(url);
 if(!Number.isInteger(limit)||limit<=0||limit>4*1024*1024)fail('INVALID_BODY_LIMIT');
 return new Promise((resolve,reject)=>{
  let ended=false,req;
  const cleanup=()=>{signal?.removeEventListener('abort',onAbort);clearTimeout(timer)};
  const finish=(error,data)=>{if(ended)return;ended=true;cleanup();if(error)reject(error);else resolve(data)};
  const onAbort=()=>req?.destroy(Object.assign(new Error('CANCELLED'),{code:'CANCELLED'}));
  const timer=setTimeout(()=>req?.destroy(Object.assign(new Error('OFFICIAL_TIMEOUT'),{code:'OFFICIAL_TIMEOUT'})),timeout);
  try{
   if(signal?.aborted){finish(Object.assign(new Error('CANCELLED'),{code:'CANCELLED'}));return}
   req=request(u,{method:'GET',agent:false,lookup:pinnedLookup,rejectUnauthorized:true,servername:u.hostname,
     headers:{'User-Agent':'Voder-OfficialDocs-Research/0.1','Accept':'text/html, application/xml, application/json, application/octet-stream, */*;q=0.1','Accept-Encoding':'identity'}},res=>{
    const status=res.statusCode??0;
    // Explicitly report redirects, do not automatically follow third-party URLs.
    if(status<200||status>=300){res.resume();finish(Object.assign(new Error('OFFICIAL_HTTP_'+status),{code:'OFFICIAL_HTTP_'+status}));return;}
    if(res.headers['content-encoding']&&res.headers['content-encoding']!=='identity'){
      res.resume();finish(Object.assign(new Error('UNSUPPORTED_CONTENT_ENCODING'),{code:'UNSUPPORTED_CONTENT_ENCODING'}));return;}
    const chunks=[];let size=0;
    res.on('data',chunk=>{size+=chunk.length;if(size>limit){req.destroy(Object.assign(new Error('OFFICIAL_RESPONSE_TOO_LARGE'),{code:'OFFICIAL_RESPONSE_TOO_LARGE'}));return}chunks.push(chunk)});
    res.on('end',()=>finish(null,{bytes:Buffer.concat(chunks),contentType:String(res.headers['content-type']??''),url:u.href,status}));
    res.on('error',e=>finish(e));
   });
   req.on('error',e=>finish(e));signal?.addEventListener('abort',onAbort,{once:true});req.end();
  }catch(e){finish(e)}
 });
}
