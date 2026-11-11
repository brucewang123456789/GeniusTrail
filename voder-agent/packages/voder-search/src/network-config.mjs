/** Host-owned DNS and connection policy, scoped to this retrieval module. */
import {checkedPublicUrl,publicAddress} from './keyless-web.mjs';
const error=()=>{throw Object.assign(new Error('WEB_CONFIG_INVALID'),{code:'WEB_CONFIG_INVALID'})};
const plain=x=>!!x&&typeof x==='object'&&!Array.isArray(x)&&[Object.prototype,null].includes(Object.getPrototypeOf(x));
const fields=(x,keys)=>{if(!plain(x)||Object.keys(x).some(k=>!keys.includes(k)))error();};
const resolver=(id,url,bootstrap_ips)=>({id,url,bootstrap_ips,method:'POST',allow_loopback:false});
export const DEFAULT_NETWORK=Object.freeze({dns:{mode:'secure',resolvers:[resolver('google','https://dns.google/dns-query',['8.8.8.8','8.8.4.4']),resolver('cloudflare','https://cloudflare-dns.com/dns-query',['1.1.1.1','1.0.0.1']),resolver('quad9','https://dns.quad9.net/dns-query',['9.9.9.9','149.112.112.112']),resolver('alidns','https://dns.alidns.com/dns-query',['223.5.5.5','223.6.6.6'])],timeout_ms:2500,cache_ttl_secs:300,require_ad:false,address_family:'auto'},connect_timeout_ms:3000});
export function normalizeNetwork(raw={},parent){
 fields(raw,['dns','connect_timeout_ms']);if(raw.dns!==undefined)fields(raw.dns,['mode','resolvers','timeout_ms','cache_ttl_secs','require_ad','address_family']);
 const base=parent??DEFAULT_NETWORK;
 const n={dns:{...base.dns,...raw.dns},connect_timeout_ms:raw.connect_timeout_ms??base.connect_timeout_ms};
 const d=n.dns;
 if(!['secure','system'].includes(d.mode)||!['auto','ipv4','ipv6'].includes(d.address_family)||typeof d.require_ad!=='boolean'||!Number.isSafeInteger(d.timeout_ms)||d.timeout_ms<250||d.timeout_ms>10000||!Number.isSafeInteger(d.cache_ttl_secs)||d.cache_ttl_secs<0||d.cache_ttl_secs>3600||!Number.isSafeInteger(n.connect_timeout_ms)||n.connect_timeout_ms<250||n.connect_timeout_ms>10000||!Array.isArray(d.resolvers)||d.resolvers.length>4||(d.mode==='secure'&&!d.resolvers.length))error();
 const seen=new Set();
 d.resolvers=d.resolvers.map(input=>{
  fields(input,['id','url','bootstrap_ips','method','allow_loopback']);const r={bootstrap_ips:[],method:'POST',allow_loopback:false,...input};
  if(typeof r.id!=='string'||!/^[-a-zA-Z0-9_]{1,64}$/.test(r.id)||seen.has(r.id)||typeof r.url!=='string'||r.url.length>1300||/[\x00-\x20\x7f]/.test(r.url)||!['GET','POST'].includes(r.method)||typeof r.allow_loopback!=='boolean'||!Array.isArray(r.bootstrap_ips)||r.bootstrap_ips.length>4)error();seen.add(r.id);
  let u;try{u=new URL(r.url);}catch{error();}
  const local=r.allow_loopback&&u.protocol==='http:'&&['127.0.0.1','[::1]'].includes(u.hostname);
  if(u.username||u.password||u.hash||u.search)error();
  if(local){if(r.bootstrap_ips.length)error();}
  else{try{const check=new URL(u.href);check.port='';checkedPublicUrl(check.href);}catch{error();}if(r.allow_loopback||r.bootstrap_ips.some(ip=>!publicAddress(ip)))error();}
  r.bootstrap_ips=[...r.bootstrap_ips];r.url=u.href;return r;
 });return n;
}
