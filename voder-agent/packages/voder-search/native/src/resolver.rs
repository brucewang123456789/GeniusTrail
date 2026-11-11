//! Module-scoped RFC8484 resolution. Never edits hosts, resolv.conf or process DNS.
use super::{
    checked_url, fail, loopback_url, proxy_applies, public_ip, validate_addresses, Failure, R,
};
use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine};
use hickory_proto::{
    op::{Edns, Message, MessageType, OpCode, Query, ResponseCode},
    rr::{DNSClass, Name, RData, RecordType},
};
use reqwest::{blocking::Client, redirect::Policy};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    collections::HashSet,
    io::Read,
    net::{IpAddr, SocketAddr, ToSocketAddrs},
    sync::mpsc,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};
use url::Url;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Resolver {
    pub id: String,
    pub url: String,
    pub bootstrap_ips: Vec<IpAddr>,
    pub method: String,
    pub allow_loopback: bool,
}
impl Default for Resolver {
    fn default() -> Self {
        Self {
            id: String::new(),
            url: String::new(),
            bootstrap_ips: vec![],
            method: "POST".into(),
            allow_loopback: false,
        }
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct DnsPolicy {
    pub mode: String,
    pub resolvers: Vec<Resolver>,
    pub timeout_ms: u64,
    pub cache_ttl_secs: u32,
    pub require_ad: bool,
    pub address_family: String,
}
impl Default for DnsPolicy {
    fn default() -> Self {
        let make = |id: &str, url: &str, ips: &[&str]| Resolver {
            id: id.into(),
            url: url.into(),
            bootstrap_ips: ips.iter().map(|s| s.parse().unwrap()).collect(),
            ..Resolver::default()
        };
        Self {
            mode: "secure".into(),
            resolvers: vec![
                make(
                    "google",
                    "https://dns.google/dns-query",
                    &["8.8.8.8", "8.8.4.4"],
                ),
                make(
                    "cloudflare",
                    "https://cloudflare-dns.com/dns-query",
                    &["1.1.1.1", "1.0.0.1"],
                ),
                make(
                    "quad9",
                    "https://dns.quad9.net/dns-query",
                    &["9.9.9.9", "149.112.112.112"],
                ),
                make(
                    "alidns",
                    "https://dns.alidns.com/dns-query",
                    &["223.5.5.5", "223.6.6.6"],
                ),
            ],
            timeout_ms: 2500,
            cache_ttl_secs: 300,
            require_ad: false,
            address_family: "auto".into(),
        }
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Network {
    pub dns: DnsPolicy,
    pub connect_timeout_ms: u64,
}
impl Default for Network {
    fn default() -> Self {
        Self {
            dns: DnsPolicy::default(),
            connect_timeout_ms: 3000,
        }
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CachedDns {
    pub host: String,
    pub policy_hash: String,
    pub expires_ms: u64,
    pub addresses: Vec<IpAddr>,
}
pub struct Resolution {
    pub addresses: Vec<SocketAddr>,
    pub receipt: Value,
    pub cache: Option<CachedDns>,
}
#[derive(Debug)]
struct Packet {
    addresses: Vec<IpAddr>,
    ttl: u32,
    ad: bool,
    rcode: u16,
}
struct Group {
    index: usize,
    addresses: Vec<IpAddr>,
    ttl: u32,
    ad: bool,
}
fn canonical(s: &str) -> String {
    s.trim_end_matches('.').to_ascii_lowercase()
}
fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}
pub fn hash(policy: &Network) -> String {
    format!(
        "{:x}",
        Sha256::digest(serde_json::to_vec(policy).unwrap_or_default())
    )
}
fn time_left(deadline: Instant) -> R<Duration> {
    let left = deadline.saturating_duration_since(Instant::now());
    if left < Duration::from_millis(10) {
        Err(fail("WEB_TIMEOUT"))
    } else {
        Ok(left)
    }
}
fn safe_id(s: &str) -> bool {
    !s.is_empty()
        && s.len() <= 64
        && s.bytes()
            .all(|c| c.is_ascii_alphanumeric() || c == b'-' || c == b'_')
}
pub fn validate(n: &Network) -> R<()> {
    let p = &n.dns;
    if !["secure", "system"].contains(&p.mode.as_str())
        || !["auto", "ipv4", "ipv6"].contains(&p.address_family.as_str())
        || !(250..=10000).contains(&p.timeout_ms)
        || p.cache_ttl_secs > 3600
        || !(250..=10000).contains(&n.connect_timeout_ms)
        || p.resolvers.len() > 4
        || (p.mode == "secure" && p.resolvers.is_empty())
    {
        return Err(fail("WEB_CONFIG_INVALID"));
    }
    let mut ids = HashSet::new();
    for r in &p.resolvers {
        if !safe_id(&r.id)
            || !ids.insert(&r.id)
            || !["GET", "POST"].contains(&r.method.as_str())
            || r.bootstrap_ips.len() > 4
        {
            return Err(fail("WEB_CONFIG_INVALID"));
        }
        let u = Url::parse(&r.url).map_err(|_| fail("WEB_CONFIG_INVALID"))?;
        let local = r.allow_loopback && loopback_url(&u);
        if local {
            if !u.username().is_empty()
                || u.password().is_some()
                || u.fragment().is_some()
                || u.query().is_some()
                || r.url.len() > 1300
                || r.url.chars().any(|c| c <= ' ' || c == '\u{7f}')
                || !r.bootstrap_ips.is_empty()
            {
                return Err(fail("WEB_CONFIG_INVALID"));
            }
        } else {
            let mut check = u.clone();
            check
                .set_port(None)
                .map_err(|_| fail("WEB_CONFIG_INVALID"))?;
            checked_url(check.as_str()).map_err(|_| fail("WEB_CONFIG_INVALID"))?;
            if u.query().is_some()
                || u.fragment().is_some()
                || r.allow_loopback
                || r.bootstrap_ips.iter().any(|ip| !public_ip(*ip))
            {
                return Err(fail("WEB_CONFIG_INVALID"));
            }
        }
    }
    Ok(())
}
fn system(host: &str, port: u16, deadline: Instant) -> R<Vec<SocketAddr>> {
    let h = host.to_owned();
    let (tx, rx) = mpsc::sync_channel(1);
    std::thread::spawn(move || {
        let _ = tx.send(
            (h.trim_matches(['[', ']']), port)
                .to_socket_addrs()
                .map(|a| a.collect()),
        );
    });
    rx.recv_timeout(time_left(deadline)?)
        .map_err(|_| fail("WEB_TIMEOUT"))?
        .map_err(|_| fail("WEB_DNS_FAILED"))
}
fn query(host: &str, kind: RecordType) -> R<Vec<u8>> {
    let mut m = Message::new(0, MessageType::Query, OpCode::Query);
    m.metadata.recursion_desired = true;
    m.add_query(Query::query(
            Name::from_ascii(host).map_err(|_| fail("WEB_DNS_FAILED"))?,
            kind,
        ));
    let mut edns = Edns::new();
    edns.set_max_payload(4096).set_dnssec_ok(true);
    m.set_edns(edns);
    m.to_vec().map_err(|_| fail("WEB_DNS_FAILED"))
}
fn parse(bytes: &[u8], host: &str, kind: RecordType, require_ad: bool) -> R<Packet> {
    if bytes.len() > 65535 {
        return Err(fail("WEB_DNS_INVALID_RESPONSE"));
    }
    let m = Message::from_vec(bytes).map_err(|_| fail("WEB_DNS_INVALID_RESPONSE"))?;
    if m.metadata.id != 0
        || m.metadata.message_type != MessageType::Response
        || m.metadata.op_code != OpCode::Query
        || m.metadata.truncation
        || m.queries.len() != 1
        || m.queries[0].query_type() != kind
        || m.queries[0].query_class() != DNSClass::IN
        || canonical(&m.queries[0].name().to_ascii()) != canonical(host)
        || m.answers.len() > 256
    {
        return Err(fail("WEB_DNS_INVALID_RESPONSE"));
    }
    if ![ResponseCode::NoError, ResponseCode::NXDomain].contains(&m.metadata.response_code) {
        return Err(fail("WEB_DNS_FAILED"));
    }
    if require_ad && !m.metadata.authentic_data {
        return Err(fail("WEB_DNSSEC_REQUIRED"));
    }
    let mut owner = canonical(host);
    let mut seen = HashSet::from([owner.clone()]);
    let mut ttl = u32::MAX;
    for _ in 0..9 {
        let aliases: Vec<_> = m
            .answers
            .iter()
            .filter(|rr| {
                rr.dns_class == DNSClass::IN && canonical(&rr.name.to_ascii()) == owner
            })
            .filter_map(|rr| {
                if let RData::CNAME(v) = &rr.data {
                    Some((canonical(&v.0.to_ascii()), rr.ttl))
                } else {
                    None
                }
            })
            .collect();
        if aliases.is_empty() {
            break;
        }
        let next = &aliases[0].0;
        if aliases.iter().any(|(n, _)| n != next) || !seen.insert(next.clone()) || seen.len() > 9 {
            return Err(fail("WEB_DNS_INVALID_RESPONSE"));
        }
        ttl = ttl.min(aliases.iter().map(|(_, t)| *t).min().unwrap());
        owner = next.clone();
    }
    let mut addresses = vec![];
    for rr in m.answers.iter().filter(|rr| {
        rr.dns_class == DNSClass::IN
            && canonical(&rr.name.to_ascii()) == owner
            && rr.record_type() == kind
    }) {
        let ip = match &rr.data {
            RData::A(v) => IpAddr::V4(v.0),
            RData::AAAA(v) => IpAddr::V6(v.0),
            _ => continue,
        };
        if !public_ip(ip) {
            return Err(fail("DNS_PUBLIC_ADDRESS_REQUIRED"));
        }
        if !addresses.contains(&ip) {
            addresses.push(ip);
        }
        ttl = ttl.min(rr.ttl);
    }
    if addresses.len() > 64 || (!addresses.is_empty() && m.metadata.response_code != ResponseCode::NoError)
    {
        return Err(fail("WEB_DNS_INVALID_RESPONSE"));
    }
    Ok(Packet {
        addresses,
        ttl: if ttl == u32::MAX { 0 } else { ttl },
        ad: m.metadata.authentic_data,
        rcode: u16::from(m.metadata.response_code),
    })
}
fn resolver_group(
    r: Resolver,
    index: usize,
    host: String,
    p: DnsPolicy,
    deadline: Instant,
) -> (Option<Group>, Value) {
    let started = Instant::now();
    let mut receipt = json!({"resolver":r.id,"transport":"RFC8484","queries":[]});
    let result = (|| -> R<Group> {
        let u = Url::parse(&r.url).map_err(|_| fail("WEB_CONFIG_INVALID"))?;
        let local = r.allow_loopback && loopback_url(&u);
        let dns_host = u.host_str().unwrap();
        let port = u.port_or_known_default().unwrap();
        if !local && proxy_applies(dns_host) {
            return Err(fail("WEB_PROXY_UNSUPPORTED"));
        }
        let ips = if local {
            vec![]
        } else if r.bootstrap_ips.is_empty() {
            validate_addresses(system(dns_host, port, deadline)?)?
                .into_iter()
                .map(|a| a.ip())
                .collect()
        } else {
            r.bootstrap_ips.clone()
        };
        receipt["resolver_host"] = json!(dns_host);
        receipt["bootstrap_ips"] = json!(ips);
        receipt["tls_validation"] = json!(if local {
            "not_applicable_loopback"
        } else {
            "required"
        });
        let bootstrap: Vec<_> = ips.iter().map(|ip| SocketAddr::new(*ip, port)).collect();
        let mut builder = Client::builder()
            .no_proxy()
            .https_only(!local)
            .redirect(Policy::none())
            .referer(false)
            .timeout(time_left(deadline)?)
            .connect_timeout(time_left(deadline)?.min(Duration::from_millis(1200)));
        if !local {
            builder = builder.resolve_to_addrs(dns_host, &bootstrap);
        }
        let client = builder.build().map_err(|_| fail("WEB_DNS_FAILED"))?;
        let kinds = match p.address_family.as_str() {
            "ipv4" => vec![RecordType::A],
            "ipv6" => vec![RecordType::AAAA],
            _ => vec![RecordType::A, RecordType::AAAA],
        };
        receipt["queries_scheduled"] = json!(kinds.len());
        let (tx, rx) = mpsc::channel();
        for kind in kinds.iter().copied() {
            let client = client.clone();
            let url = u.clone();
            let h = host.clone();
            let tx = tx.clone();
            let method = r.method.clone();
            let ad = p.require_ad;
            std::thread::spawn(move || {
                let out = (|| -> R<(Packet, Value)> {
                    let wire = query(&h, kind)?;
                    let req = if method == "GET" {
                        client
                            .get(url)
                            .query(&[("dns", URL_SAFE_NO_PAD.encode(&wire))])
                    } else {
                        client
                            .post(url)
                            .header("Content-Type", "application/dns-message")
                            .body(wire)
                    };
                    let mut response = req
                        .header("Accept", "application/dns-message")
                        .header("Accept-Encoding", "identity")
                        .header("User-Agent", "Voder-Retrieval-DNS/1.0")
                        .timeout(time_left(deadline)?)
                        .send()
                        .map_err(|e| {
                            fail(if e.is_timeout() {
                                "WEB_TIMEOUT"
                            } else {
                                "WEB_DNS_FAILED"
                            })
                        })?;
                    let status = response.status().as_u16();
                    let peer = response.remote_addr().map(|a| a.ip().to_string());
                    if status != 200 {
                        return Err(fail("WEB_DNS_FAILED"));
                    }
                    let ct = response
                        .headers()
                        .get("content-type")
                        .and_then(|v| v.to_str().ok())
                        .unwrap_or("")
                        .split(';')
                        .next()
                        .unwrap_or("")
                        .trim();
                    if ct != "application/dns-message"
                        || response
                            .headers()
                            .get("content-encoding")
                            .is_some_and(|v| v != "identity")
                    {
                        return Err(fail("WEB_DNS_INVALID_RESPONSE"));
                    }
                    let mut body = vec![];
                    response
                        .by_ref()
                        .take(65536)
                        .read_to_end(&mut body)
                        .map_err(|_| fail("WEB_DNS_FAILED"))?;
                    let mut packet = parse(&body, &h, kind, ad)?;
                    let age = response
                        .headers()
                        .get("age")
                        .map(|v| {
                            v.to_str()
                                .ok()
                                .and_then(|s| s.parse::<u32>().ok())
                                .ok_or_else(|| fail("WEB_DNS_INVALID_RESPONSE"))
                        })
                        .transpose()?
                        .unwrap_or(0);
                    packet.ttl = packet.ttl.saturating_sub(age);
                    let info = json!({"type":u16::from(kind),"rcode":packet.rcode,"resolver_ad":packet.ad,"ttl":packet.ttl,"addresses":packet.addresses,"peer_ip":peer,"http_status":status,"tls_validated":!local});
                    Ok((packet, info))
                })();
                let _ = tx.send((kind, out));
            });
        }
        drop(tx);
        let mut addresses = vec![];
        let mut ttl = u32::MAX;
        let mut all_ad = true;
        let mut errors = vec![];
        let mut completed = 0;
        while completed < kinds.len() {
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining < Duration::from_millis(10) {
                break;
            }
            match rx.recv_timeout(remaining) {
                Ok((_kind, Ok((packet, info)))) => {
                    completed += 1;
                    receipt["queries"].as_array_mut().unwrap().push(info);
                    if !packet.addresses.is_empty() {
                        ttl = ttl.min(packet.ttl);
                    }
                    all_ad &= packet.ad;
                    addresses.extend(packet.addresses);
                }
                Ok((kind, Err(e))) => {
                    completed += 1;
                    receipt["queries"]
                        .as_array_mut()
                        .unwrap()
                        .push(json!({"type":u16::from(kind),"code":e.code}));
                    errors.push(e.code);
                }
                Err(_) => break,
            }
        }
        for kind in &kinds {
            if !receipt["queries"]
                .as_array()
                .unwrap()
                .iter()
                .any(|q| q["type"] == json!(u16::from(*kind)))
            {
                receipt["queries"].as_array_mut().unwrap().push(json!({"type":u16::from(*kind),"code":"WEB_TIMEOUT","response_before_deadline":false}));
            }
        }
        // An explicitly unsafe DNS answer invalidates this resolver, never just one family.
        if errors.contains(&"DNS_PUBLIC_ADDRESS_REQUIRED") {
            return Err(fail("DNS_PUBLIC_ADDRESS_REQUIRED"));
        }
        if p.require_ad && (completed != kinds.len() || !errors.is_empty()) {
            return Err(fail("WEB_DNSSEC_REQUIRED"));
        }
        if addresses.is_empty() {
            return Err(fail(if errors.contains(&"WEB_DNSSEC_REQUIRED") {
                "WEB_DNSSEC_REQUIRED"
            } else {
                "WEB_DNS_FAILED"
            }));
        }
        Ok(Group {
            index,
            addresses,
            ttl: if ttl == u32::MAX { 0 } else { ttl },
            ad: all_ad && errors.is_empty() && completed == kinds.len(),
        })
    })();
    receipt["elapsed_ms"] = json!(started.elapsed().as_millis());
    match result {
        Ok(group) => {
            receipt["status"] = json!("RESOLVED");
            (Some(group), receipt)
        }
        Err(e) => {
            receipt["status"] = json!("FAILED");
            receipt["code"] = json!(e.code);
            (None, receipt)
        }
    }
}
fn family(addresses: Vec<IpAddr>, family: &str, port: u16) -> Vec<SocketAddr> {
    let mut seen = HashSet::new();
    let mut v: Vec<_> = addresses
        .into_iter()
        .filter(|ip| match family {
            "ipv4" => ip.is_ipv4(),
            "ipv6" => ip.is_ipv6(),
            _ => true,
        })
        .filter(|ip| seen.insert(*ip))
        .map(|ip| SocketAddr::new(ip, port))
        .collect();
    v.sort_by_key(|a| if a.is_ipv4() { 0 } else { 1 });
    v
}
pub fn resolve(
    host: &str,
    port: u16,
    n: &Network,
    deadline: Instant,
    local: bool,
    cached: Option<&CachedDns>,
) -> R<Resolution> {
    validate(n)?;
    if local {
        return Ok(Resolution {
            addresses: system(host, port, deadline)?,
            receipt: json!({"mode":"LOOPBACK_LITERAL","cache_hit":false,"queries_scheduled":0}),
            cache: None,
        });
    }
    let fingerprint = hash(n);
    if let Some(c) = cached {
        if c.host == host
            && c.policy_hash == fingerprint
            && c.expires_ms > now_ms()
            && c.expires_ms <= now_ms() + u64::from(n.dns.cache_ttl_secs) * 1000
            && !c.addresses.is_empty()
            && c.addresses.len() <= 64
            && c.addresses.iter().all(|ip| public_ip(*ip))
        {
            let addresses = family(c.addresses.clone(), &n.dns.address_family, port);
            if !addresses.is_empty() {
                return Ok(Resolution {
                    addresses,
                    receipt: json!({"mode":n.dns.mode,"cache_hit":true,"queries_scheduled":0,"expires_ms":c.expires_ms}),
                    cache: Some(c.clone()),
                });
            }
        }
    }
    if n.dns.mode == "system" {
        let addresses = validate_addresses(system(host, port, deadline)?)?;
        let addresses = family(
            addresses.iter().map(|a| a.ip()).collect(),
            &n.dns.address_family,
            port,
        );
        if addresses.is_empty() {
            return Err(fail("WEB_DNS_FAILED"));
        }
        return Ok(Resolution {
            addresses,
            receipt: json!({"mode":"SYSTEM_EXPLICIT","cache_hit":false,"queries_scheduled":0}),
            cache: None,
        });
    }
    let limit = deadline.min(Instant::now() + Duration::from_millis(n.dns.timeout_ms));
    let (tx, rx) = mpsc::channel();
    for (index, r) in n.dns.resolvers.iter().cloned().enumerate() {
        let tx = tx.clone();
        let h = host.to_owned();
        let p = n.dns.clone();
        let worker_limit = limit - Duration::from_millis(30);
        std::thread::spawn(move || {
            let _ = tx.send(resolver_group(r, index, h, p, worker_limit));
        });
    }
    drop(tx);
    let mut groups = vec![];
    let mut receipts = vec![];
    for _ in 0..n.dns.resolvers.len() {
        let left = limit.saturating_duration_since(Instant::now());
        if left.is_zero() {
            break;
        }
        match rx.recv_timeout(left) {
            Ok((g, r)) => {
                receipts.push(r);
                if let Some(g) = g {
                    groups.push(g);
                }
            }
            Err(_) => break,
        }
    }
    receipts.sort_by_key(|r| {
        n.dns
            .resolvers
            .iter()
            .position(|p| Some(p.id.as_str()) == r["resolver"].as_str())
            .unwrap_or(usize::MAX)
    });
    if groups.is_empty() {
        return Err(Failure {
            code: "WEB_SECURE_DNS_UNAVAILABLE",
            status: None,
            trace: json!({"host":host,"stage":"SECURE_DNS","mode":"secure","cache_hit":false,"resolvers":receipts}),
        });
    }
    // Prefer authenticated-data answers, then explicit operator resolver order. Distinct
    // public answers are retained as TLS-validated connection alternatives (CDNs vary).
    if groups.iter().any(|g| g.ad) {
        groups.retain(|g| g.ad);
    }
    groups.sort_by_key(|g| (!g.ad, g.index));
    let ttl = groups
        .iter()
        .map(|g| g.ttl)
        .min()
        .unwrap_or(0)
        .min(n.dns.cache_ttl_secs);
    let mut alternatives = vec![];
    for offset in 0..64 {
        for group in &groups {
            if let Some(ip) = group.addresses.get(offset) {
                alternatives.push(*ip);
            }
        }
    }
    let addresses = family(alternatives, &n.dns.address_family, port);
    if addresses.is_empty() || addresses.len() > 64 {
        return Err(fail("WEB_DNS_FAILED"));
    }
    let cache = Some(CachedDns {
        host: host.into(),
        policy_hash: fingerprint,
        expires_ms: now_ms() + u64::from(ttl) * 1000,
        addresses: addresses.iter().map(|a| a.ip()).collect(),
    });
    Ok(Resolution {
        addresses,
        receipt: json!({"mode":"SECURE_DOH","cache_hit":false,"queries_scheduled":receipts.iter().map(|r|r["queries_scheduled"].as_u64().unwrap_or(0)).sum::<u64>(),"resolvers":receipts,"ttl":ttl}),
        cache,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use hickory_proto::rr::{
        rdata::{A, AAAA, CNAME},
        Record,
    };
    fn response(host: &str, kind: RecordType) -> Message {
        let mut m = Message::new(0, MessageType::Response, OpCode::Query);
        m.metadata.recursion_desired = true;
        m.metadata.recursion_available = true;
        m.add_query(Query::query(Name::from_ascii(host).unwrap(), kind));
        m
    }
    #[test]
    fn standards_query_is_wire_not_json() {
        let m = Message::from_vec(&query("api.search.brave.com", RecordType::A).unwrap()).unwrap();
        assert_eq!(m.queries[0].query_type(), RecordType::A);
        assert!(m.metadata.recursion_desired);
        assert!(m.edns.as_ref().unwrap().flags().dnssec_ok);
    }
    #[test]
    fn reject_question_mismatch_and_wrong_message_type() {
        let m = response("wrong.example.org", RecordType::A);
        assert!(parse(
            &m.to_vec().unwrap(),
            "right.example.org",
            RecordType::A,
            false
        )
        .is_err());
        let q = query("right.example.org", RecordType::A).unwrap();
        assert!(parse(&q, "right.example.org", RecordType::A, false).is_err());
    }
    #[test]
    fn reject_truncation_and_invalid_wire() {
        let mut m = response("x.example.org", RecordType::A);
        m.metadata.truncation = true;
        assert!(parse(&m.to_vec().unwrap(), "x.example.org", RecordType::A, false).is_err());
        assert!(parse(&[0, 1, 2], "x.example.org", RecordType::A, false).is_err());
    }
    #[test]
    fn follow_cname_and_bind_answer_owner() {
        let mut m = response("x.example.org", RecordType::A);
        m.add_answer(Record::from_rdata(
            Name::from_ascii("x.example.org").unwrap(),
            30,
            RData::CNAME(CNAME(Name::from_ascii("edge.example.net").unwrap())),
        ));
        m.add_answer(Record::from_rdata(
            Name::from_ascii("edge.example.net").unwrap(),
            60,
            RData::A(A("8.8.8.8".parse().unwrap())),
        ));
        m.add_answer(Record::from_rdata(
            Name::from_ascii("other.example.net").unwrap(),
            60,
            RData::A(A("127.0.0.1".parse().unwrap())),
        ));
        let p = parse(&m.to_vec().unwrap(), "x.example.org", RecordType::A, false).unwrap();
        assert_eq!(p.ttl, 30);
        assert_eq!(p.addresses[0], "8.8.8.8".parse::<IpAddr>().unwrap());
    }
    #[test]
    fn reject_cname_loop() {
        let mut m = response("x.example.org", RecordType::A);
        m.add_answer(Record::from_rdata(
            Name::from_ascii("x.example.org").unwrap(),
            30,
            RData::CNAME(CNAME(Name::from_ascii("x.example.org").unwrap())),
        ));
        assert!(parse(&m.to_vec().unwrap(), "x.example.org", RecordType::A, false).is_err());
    }
    #[test]
    fn reject_mixed_private_address_answer() {
        let mut m = response("x.example.org", RecordType::A);
        for ip in ["8.8.8.8", "127.0.0.1"] {
            m.add_answer(Record::from_rdata(
                Name::from_ascii("x.example.org").unwrap(),
                30,
                RData::A(A(ip.parse().unwrap())),
            ));
        }
        assert_eq!(
            parse(&m.to_vec().unwrap(), "x.example.org", RecordType::A, false)
                .unwrap_err()
                .code,
            "DNS_PUBLIC_ADDRESS_REQUIRED"
        );
    }
    #[test]
    fn dnssec_flag_is_explicit_not_assumed() {
        let mut m = response("x.example.org", RecordType::AAAA);
        m.add_answer(Record::from_rdata(
            Name::from_ascii("x.example.org").unwrap(),
            20,
            RData::AAAA(AAAA("2606:4700:4700::1111".parse().unwrap())),
        ));
        assert!(parse(
            &m.to_vec().unwrap(),
            "x.example.org",
            RecordType::AAAA,
            true
        )
        .is_err());
        m.metadata.authentic_data = true;
        assert!(
            parse(
                &m.to_vec().unwrap(),
                "x.example.org",
                RecordType::AAAA,
                true
            )
            .unwrap()
            .ad
        );
    }
    #[test]
    fn empty_aaaa_is_valid_nodata() {
        let m = response("x.example.org", RecordType::AAAA);
        assert!(parse(
            &m.to_vec().unwrap(),
            "x.example.org",
            RecordType::AAAA,
            false
        )
        .unwrap()
        .addresses
        .is_empty());
    }
    #[test]
    fn defaults_do_not_pin_a_search_service() {
        let s = serde_json::to_string(&Network::default()).unwrap();
        assert!(!s.contains("3.33.153.106"));
        assert!(!s.contains("15.197.138.111"));
        assert!(!s.contains("api.search.brave.com"));
        validate(&Network::default()).unwrap();
    }
    #[test]
    fn bad_bootstrap_and_unknown_dns_mode_are_denied() {
        let mut n = Network::default();
        n.dns.resolvers[0].bootstrap_ips = vec!["127.0.0.1".parse().unwrap()];
        assert!(validate(&n).is_err());
        n = Network::default();
        n.dns.mode = "fallback-to-unsafe".into();
        assert!(validate(&n).is_err());
    }
}
