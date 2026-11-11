//! One bounded request per invocation. No model calls, hidden provider retries,
//! redirects, cookies, script execution, or remote Fetch API. Node owns hop policy.
mod resolver;
use reqwest::{blocking::Client, redirect::Policy};
use scraper::{Html, Selector};
use serde::Deserialize;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, HashSet},
    io::{self, Read},
    net::{IpAddr, SocketAddr},
    time::{Duration, Instant},
};
use url::Url;

const MAX_PAGE: usize = 600 * 1024;
const MAX_API: usize = 200 * 1024;
#[derive(Deserialize, Default)]
#[serde(deny_unknown_fields)]
struct Job {
    op: String,
    #[serde(default)]
    url: String,
    #[serde(default)]
    query: String,
    #[serde(default)]
    provider: String,
    #[serde(default)]
    key: String,
    #[serde(default = "default_timeout")]
    timeout_ms: u64,
    #[serde(default)]
    method: String,
    #[serde(default)]
    headers: BTreeMap<String, String>,
    #[serde(default)]
    params: BTreeMap<String, Value>,
    #[serde(default)]
    body: Option<Value>,
    #[serde(default)]
    body_format: String,
    #[serde(default)]
    response_format: String,
    #[serde(default)]
    allow_loopback: bool,
    #[serde(default)]
    network: resolver::Network,
    #[serde(default)]
    cached_dns: Option<resolver::CachedDns>,
}
fn default_timeout() -> u64 {
    12000
}
#[derive(Debug)]
struct Failure {
    code: &'static str,
    status: Option<u16>,
    trace: Value,
}
type R<T> = Result<T, Failure>;
fn fail(code: &'static str) -> Failure {
    Failure {
        code,
        status: None,
        trace: Value::Null,
    }
}
fn cut(s: &str, n: usize) -> String {
    s.chars().take(n).collect()
}
fn secret(s: &str) -> bool {
    static RE: std::sync::OnceLock<regex::Regex> = std::sync::OnceLock::new();
    RE.get_or_init(||regex::Regex::new(r"(?i)(?:api[_-]?key|access[_-]?token|auth[_-]?token|password|secret)\s*[:=]\s*\S{8,}|\bsk-[A-Za-z0-9_-]{16,}|\bgh[opurs]_[A-Za-z0-9]{16,}").unwrap()).is_match(s)
}
fn checked_url(raw: &str) -> R<Url> {
    if raw.len() > 1300 || raw.chars().any(|c| c <= ' ' || c == '\u{7f}') || secret(raw) {
        return Err(fail("UNSAFE_SOURCE_URL"));
    }
    let mut u = Url::parse(raw).map_err(|_| fail("UNSAFE_SOURCE_URL"))?;
    let h = u.host_str().unwrap_or("");
    if u.scheme() != "https"
        || !u.username().is_empty()
        || u.password().is_some()
        || u.port().is_some()
        || h.parse::<IpAddr>().is_ok()
        || !h.contains('.')
        || h.len() > 253
        || h.split('.').any(|p| {
            p.is_empty()
                || p.len() > 63
                || p.starts_with('-')
                || p.ends_with('-')
                || !p.bytes().all(|c| c.is_ascii_alphanumeric() || c == b'-')
        })
        || [
            "localhost",
            "local",
            "internal",
            "localdomain",
            "invalid",
            "onion",
        ]
        .iter()
        .any(|s| h == *s || h.ends_with(&format!(".{s}")))
    {
        return Err(fail("UNSAFE_SOURCE_URL"));
    }
    for (k, v) in u.query_pairs() {
        if [
            "apikey",
            "api_key",
            "token",
            "access_token",
            "accesstoken",
            "auth_token",
            "authtoken",
            "secret",
            "password",
            "credential",
            "authorization",
        ]
        .contains(&k.to_lowercase().as_str())
            || secret(&v)
        {
            return Err(fail("UNSAFE_SOURCE_URL"));
        }
    }
    u.set_fragment(None);
    Ok(u)
}
fn public_ip(ip: IpAddr) -> bool {
    match ip {
        IpAddr::V4(a) => {
            let n = u32::from(a);
            ![
                (0, 8),
                (0x0a000000, 8),
                (0x64400000, 10),
                (0x7f000000, 8),
                (0xa9fe0000, 16),
                (0xac100000, 12),
                (0xc0000000, 24),
                (0xc0000200, 24),
                (0xc0a80000, 16),
                (0xc6120000, 15),
                (0xc6336400, 24),
                (0xcb007100, 24),
                (0xe0000000, 4),
                (0xf0000000, 4),
            ]
            .iter()
            .any(|(b, p)| n >> (32 - p) == b >> (32 - p))
        }
        IpAddr::V6(a) => {
            let s = a.segments();
            (s[0] & 0xe000) == 0x2000
                && !(s[0] == 0x2001 && (s[1] == 0 || s[1] == 0xdb8))
                && s[0] != 0x2002
        }
    }
}
fn validate_addresses(addrs: Vec<SocketAddr>) -> R<Vec<SocketAddr>> {
    // Whole answer must pass; do not hide a private/tunnel answer by picking IPv4.
    if addrs.is_empty() || addrs.len() > 64 || addrs.iter().any(|a| !public_ip(a.ip())) {
        return Err(fail("DNS_PUBLIC_ADDRESS_REQUIRED"));
    }
    let mut seen = HashSet::new();
    let mut out: Vec<_> = addrs.into_iter().filter(|a| seen.insert(*a)).collect();
    // Native IPv6 remains available; IPv4 is preferred per request, never globally.
    out.sort_by_key(|a| if a.is_ipv4() { 0 } else { 1 });
    Ok(out)
}
fn loopback_url(u: &Url) -> bool {
    u.scheme() == "http" && ["127.0.0.1", "[::1]"].contains(&u.host_str().unwrap_or(""))
}
fn custom_spec(j: &Job) -> R<(String, Option<Value>)> {
    let mut url = if j.allow_loopback {
        let u = Url::parse(&j.url).map_err(|_| fail("UNSAFE_SOURCE_URL"))?;
        if !loopback_url(&u)
            || !u.username().is_empty()
            || u.password().is_some()
            || u.fragment().is_some()
            || j.url.len() > 1300
            || j.url.chars().any(|c| c <= ' ' || c == '\u{7f}')
        {
            return Err(fail("UNSAFE_SOURCE_URL"));
        }
        u
    } else {
        if j.url.len() > 1300 || j.url.chars().any(|c| c <= ' ' || c == '\u{7f}') || secret(&j.url)
        {
            return Err(fail("UNSAFE_SOURCE_URL"));
        }
        let mut u = Url::parse(&j.url).map_err(|_| fail("UNSAFE_SOURCE_URL"))?;
        let port = u.port();
        u.set_port(None).map_err(|_| fail("UNSAFE_SOURCE_URL"))?;
        checked_url(u.as_str())?;
        u.set_port(port).map_err(|_| fail("UNSAFE_SOURCE_URL"))?;
        u.set_fragment(None);
        u
    };
    if !["GET", "POST"].contains(&j.method.as_str())
        || !["json", "form", "text"].contains(&j.body_format.as_str())
        || !["json", "text"].contains(&j.response_format.as_str())
        || (j.method == "GET" && j.body.is_some())
        || j.headers.len() > 24
        || j.params.len() > 64
    {
        return Err(fail("INVALID_SEARCH_PROVIDER"));
    }
    for (name, value) in &j.headers {
        if [
            "host",
            "content-length",
            "connection",
            "transfer-encoding",
            "proxy-authorization",
            "proxy-connection",
            "accept-encoding",
            "user-agent",
        ]
        .contains(&name.to_lowercase().as_str())
            || name.len() > 80
            || value.len() > 8192
            || value.chars().any(char::is_control)
            || reqwest::header::HeaderName::from_bytes(name.as_bytes()).is_err()
            || reqwest::header::HeaderValue::from_str(value).is_err()
        {
            return Err(fail("INVALID_SEARCH_PROVIDER"));
        }
    }
    for (key, value) in &j.params {
        if key.len() > 128 || key.chars().any(char::is_control) {
            return Err(fail("INVALID_SEARCH_PROVIDER"));
        }
        let value = match value {
            Value::String(v) => v.clone(),
            Value::Number(n) => n.to_string(),
            Value::Bool(b) => b.to_string(),
            _ => return Err(fail("INVALID_SEARCH_PROVIDER")),
        };
        if value.len() > 8192 {
            return Err(fail("INVALID_SEARCH_PROVIDER"));
        }
        // Only fixed, host-configured API endpoints may carry query credentials.
        // Encoding values cannot change the hostname/path. No URL is included in errors.
        url.query_pairs_mut().append_pair(key, &value);
    }
    if url.as_str().len() > 32768 || j.body.as_ref().is_some_and(|b| b.to_string().len() > 65536) {
        return Err(fail("INVALID_SEARCH_PROVIDER"));
    }
    if j.body_format == "form"
        && j.body.as_ref().is_some_and(|b| {
            !b.is_object()
                || b.as_object()
                    .unwrap()
                    .values()
                    .any(|v| v.is_array() || v.is_object() || v.is_null())
        })
    {
        return Err(fail("INVALID_SEARCH_PROVIDER"));
    }
    if j.body_format == "text" && !j.body.as_ref().is_some_and(Value::is_string) {
        return Err(fail("INVALID_SEARCH_PROVIDER"));
    }
    Ok((url.into(), j.body.clone()))
}
fn request_spec(j: &Job) -> R<(String, Option<Value>)> {
    match j.op.as_str() {
        "search_custom" => custom_spec(j),
        "fetch" | "resolve" => Ok((checked_url(&j.url)?.into(), None)),
        "search" => {
            let q = j.query.trim();
            if q.is_empty()
                || q.chars().count() > 180
                || q.chars().any(char::is_control)
                || secret(q)
            {
                return Err(fail("INVALID_SEARCH_QUERY"));
            }
            if j.key.is_empty() || j.key.len() > 4096 || j.key.chars().any(char::is_control) {
                return Err(fail("WEB_CREDENTIAL_MISSING"));
            }
            match j.provider.as_str() {
                "bigmodel" => Ok((
                    "https://open.bigmodel.cn/api/paas/v4/web_search".into(),
                    Some(
                        json!({"search_query":q,"search_engine":"search_pro","count":8,"search_intent":false,"search_recency_filter":"noLimit"}),
                    ),
                )),
                "deepseek" if q.chars().count() <= 128 => Ok((
                    "https://api.deepseek.com/v1/tools/search".into(),
                    Some(json!({"query":q,"count":8,"need_content":false})),
                )),
                "deepseek" => Err(fail("INVALID_SEARCH_QUERY")),
                _ => Err(fail("INVALID_SEARCH_PROVIDER")),
            }
        }
        _ => Err(fail("INVALID_SEARCH_PROVIDER")),
    }
}
fn proxy_applies(host: &str) -> bool {
    let get = |lo: &str, up: &str| {
        std::env::var(lo)
            .ok()
            .filter(|s| !s.is_empty())
            .or_else(|| std::env::var(up).ok().filter(|s| !s.is_empty()))
    };
    let proxy = get("https_proxy", "HTTPS_PROXY").or_else(|| get("all_proxy", "ALL_PROXY"));
    if proxy.is_none() {
        return false;
    }
    let bypass = get("no_proxy", "NO_PROXY").unwrap_or_default();
    !bypass.split(',').map(str::trim).any(|s| {
        s == "*"
            || (!s.is_empty()
                && (host == s.trim_start_matches('.')
                    || host.ends_with(&format!(".{}", s.trim_start_matches('.')))))
    })
}
fn normalize(provider: &str, value: &Value) -> R<Value> {
    let a = value
        .get(if provider == "bigmodel" {
            "search_result"
        } else {
            "results"
        })
        .and_then(Value::as_array)
        .ok_or_else(|| fail("WEB_PROVIDER_INVALID"))?;
    if a.len() > 50 {
        return Err(fail("WEB_PROVIDER_INVALID"));
    }
    let mut seen = HashSet::new();
    let mut rows = vec![];
    for row in a.iter().take(16) {
        let url = row
            .get(if provider == "bigmodel" {
                "link"
            } else {
                "url"
            })
            .and_then(Value::as_str)
            .ok_or_else(|| fail("WEB_PROVIDER_INVALID"))?;
        let title = row
            .get("title")
            .and_then(Value::as_str)
            .ok_or_else(|| fail("WEB_PROVIDER_INVALID"))?;
        let excerpt = row
            .get(if provider == "bigmodel" {
                "content"
            } else {
                "summary"
            })
            .and_then(Value::as_str)
            .or_else(|| row.get("snippet").and_then(Value::as_str))
            .ok_or_else(|| fail("WEB_PROVIDER_INVALID"))?;
        if let Ok(url) = checked_url(url) {
            if seen.insert(url.to_string()) {
                rows.push(json!({"url":url.as_str(),"title":cut(title,160),"excerpt":cut(excerpt,1200),"provider":provider,"evidence_kind":"SEARCH_SNIPPET"}));
            }
        }
    }
    Ok(json!(rows))
}
fn extract(raw: &[u8], ct: &str) -> R<String> {
    let text = std::str::from_utf8(raw).map_err(|_| fail("WEB_EXTRACTION_FAILED"))?;
    if !ct.contains("html") {
        let out = cut(text.trim(), 350000);
        return if out.is_empty() {
            Err(fail("WEB_EXTRACTION_FAILED"))
        } else {
            Ok(out)
        };
    }
    let dom = Html::parse_document(text);
    let body = Selector::parse("body").unwrap();
    let root = dom.select(&body).next().unwrap_or(dom.root_element());
    let mut parts = Vec::new();
    let mut chars = 0;
    for node in root.descendants() {
        if let Some(t) = node.value().as_text() {
            let blocked = node.ancestors().any(|a| {
                a.value().as_element().is_some_and(|e| {
                    [
                        "script", "style", "noscript", "svg", "iframe", "template", "form", "nav",
                        "footer", "header",
                    ]
                    .contains(&e.name())
                })
            });
            if !blocked {
                let clean = t.split_whitespace().collect::<Vec<_>>().join(" ");
                chars += clean.len();
                parts.push(clean);
                if chars > 400000 {
                    break;
                }
            }
        }
    }
    let out = cut(parts.join(" ").trim(), 350000);
    if out.is_empty() {
        Err(fail("WEB_EXTRACTION_FAILED"))
    } else {
        Ok(out)
    }
}
fn perform(j: Job) -> R<Value> {
    if !(500..=30000).contains(&j.timeout_ms) {
        return Err(fail("INVALID_SEARCH_POLICY"));
    }
    let started = Instant::now();
    let (raw, body) = request_spec(&j)?;
    // custom_spec already validated the endpoint before adding encoded API parameters.
    let u = if j.op == "search_custom" {
        Url::parse(&raw).map_err(|_| fail("INVALID_SEARCH_PROVIDER"))?
    } else {
        checked_url(&raw)?
    };
    let host = u.host_str().unwrap();
    let local = j.op == "search_custom" && j.allow_loopback && loopback_url(&u);
    let port = u
        .port_or_known_default()
        .ok_or_else(|| fail("UNSAFE_SOURCE_URL"))?;
    if j.op != "resolve" && !local && proxy_applies(host) {
        return Err(Failure {
            code: "WEB_PROXY_UNSUPPORTED",
            status: None,
            trace: json!({"host":host,"stage":"PROXY_POLICY","result":"UNSUPPORTED_APPLICABLE_PROXY"}),
        });
    }
    let deadline = started + Duration::from_millis(j.timeout_ms);
    let resolution = resolver::resolve(
        host,
        port,
        &j.network,
        deadline,
        local,
        j.cached_dns.as_ref(),
    )?;
    let addrs = resolution.addresses;
    let mut trace = json!({"host":host,"dns_addresses":addrs.iter().map(|a|a.ip().to_string()).collect::<Vec<_>>(),"dns":resolution.receipt,"dns_cache":resolution.cache,"attempts":[],"proxy":"DIRECT","tls_validation":if local {"not_applicable_loopback"}else{"required"},"automatic_redirects":false});
    if j.op == "resolve" {
        return Ok(json!({"network_receipt":trace}));
    }
    let limit = if j.op == "fetch" { MAX_PAGE } else { MAX_API };
    let mut last = fail("WEB_CONNECT_FAILED");
    let mut choices: Vec<_> = addrs.iter().take(6).copied().collect();
    if !choices.iter().any(SocketAddr::is_ipv6) {
        if let Some(v6) = addrs.iter().find(|a| a.is_ipv6()) {
            if choices.len() == 6 {
                choices.pop();
            }
            choices.push(*v6);
        }
    }
    for addr in &choices {
        let left = Duration::from_millis(j.timeout_ms).saturating_sub(started.elapsed());
        if left < Duration::from_millis(100) {
            last = fail("WEB_TIMEOUT");
            break;
        }
        let client = Client::builder()
            .https_only(!local)
            .no_proxy()
            .redirect(Policy::none())
            .referer(false)
            .connect_timeout(left.min(Duration::from_millis(j.network.connect_timeout_ms)))
            .timeout(left)
            .resolve_to_addrs(host, &[*addr])
            .build()
            .map_err(|_| fail("WEB_PROVIDER_FAILED"))?;
        let mut req = if j.op == "search_custom" {
            let mut request = if j.method == "POST" {
                client.post(u.clone())
            } else {
                client.get(u.clone())
            };
            for (name, value) in &j.headers {
                request = request.header(name.as_str(), value.as_str());
            }
            if let Some(ref b) = body {
                if j.body_format == "text" {
                    request = request.body(b.as_str().unwrap().to_owned());
                } else if j.body_format == "form" {
                    let pairs: BTreeMap<_, _> = b
                        .as_object()
                        .unwrap()
                        .iter()
                        .map(|(k, v)| {
                            (
                                k.clone(),
                                v.as_str()
                                    .map(str::to_owned)
                                    .unwrap_or_else(|| v.to_string()),
                            )
                        })
                        .collect();
                    request = request.form(&pairs);
                } else {
                    request = request.json(b);
                }
            }
            request
        } else if let Some(ref b) = body {
            client.post(u.clone()).bearer_auth(&j.key).json(b)
        } else {
            client.get(u.clone())
        };
        let accept = if j.op == "search_custom" && j.response_format == "text" {
            "application/xml, text/xml, text/plain, text/csv, application/json"
        } else if j.op == "fetch" {
            "text/html, text/plain, text/markdown, application/xhtml+xml"
        } else {
            "application/json"
        };
        if !j.headers.keys().any(|k| k.eq_ignore_ascii_case("accept")) {
            req = req.header("Accept", accept);
        }
        req = req
            .header("Accept-Encoding", "identity")
            .header("User-Agent", "Voder-Retrieval/1.0");
        let mut response = match req.send() {
            Ok(r) => r,
            Err(e) => {
                let mut source: Option<&(dyn std::error::Error + 'static)> = Some(&e);
                let mut tls_error = false;
                while let Some(cause) = source {
                    if cause.downcast_ref::<rustls::Error>().is_some() {
                        tls_error = true;
                        break;
                    }
                    source = cause.source();
                }
                let mut os_error = None;
                let mut error_kind = None;
                let mut source: Option<&(dyn std::error::Error + 'static)> = Some(&e);
                while let Some(cause) = source {
                    if let Some(io) = cause.downcast_ref::<io::Error>() {
                        os_error = io.raw_os_error();
                        error_kind = Some(format!("{:?}", io.kind()));
                        if os_error.is_some() {
                            break;
                        }
                    }
                    source = cause.source();
                }
                let code = if tls_error {
                    "WEB_TLS_FAILED"
                } else if e.is_timeout() {
                    "WEB_TIMEOUT"
                } else if e.is_connect() && !tls_error {
                    "WEB_CONNECT_FAILED"
                } else {
                    "WEB_RESPONSE_INCOMPLETE"
                };
                trace["attempts"]
                    .as_array_mut()
                    .unwrap()
                    .push(json!({"selected_ip":addr.ip().to_string(),"result":code,"os_error":os_error,"error_kind":error_kind,"stage":if tls_error{"TLS"}else{"CONNECT_OR_HEADERS"},"retry_before_request":e.is_connect()||tls_error,"elapsed_ms":started.elapsed().as_millis()}));
                last = fail(code);
                if e.is_connect() || tls_error {
                    continue;
                }
                last.trace = trace;
                return Err(last);
            }
        };
        let status = response.status().as_u16();
        trace["attempts"].as_array_mut().unwrap().push(json!({"selected_ip":addr.ip().to_string(),"peer_ip":response.remote_addr().map(|x|x.ip().to_string()),"result":"HTTP_RESPONSE","http_status":status,"tls_validated":!local}));
        // Credential-bearing API calls may never redirect to another endpoint.
        if [301, 302, 303, 307, 308].contains(&status) {
            if j.op != "fetch" {
                let mut e = fail("WEB_REDIRECT_DENIED");
                e.trace = trace;
                return Err(e);
            }
            let location = response
                .headers()
                .get("location")
                .and_then(|v| v.to_str().ok())
                .ok_or_else(|| fail("WEB_REDIRECT_DENIED"))?;
            let next = u.join(location).map_err(|_| fail("WEB_REDIRECT_DENIED"))?;
            let next = checked_url(next.as_str())?;
            return Ok(json!({"redirect":next.as_str(),"status":status,"network_receipt":trace}));
        }
        if !(200..300).contains(&status) {
            let code = match status {
                401 => "WEB_AUTH_FAILED",
                403 => "WEB_ACCESS_DENIED",
                404 | 410 => "WEB_SOURCE_NOT_FOUND",
                429 => "WEB_RATE_LIMITED",
                _ => "WEB_HTTP_ERROR",
            };
            return Err(Failure {
                code,
                status: Some(status),
                trace,
            });
        }
        let ct = response
            .headers()
            .get("content-type")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("")
            .split(';')
            .next()
            .unwrap_or("")
            .trim()
            .to_lowercase();
        let valid = if j.op == "fetch" {
            [
                "text/html",
                "application/xhtml+xml",
                "text/plain",
                "text/markdown",
            ]
            .contains(&ct.as_str())
        } else {
            ct == "application/json"
                || ct.ends_with("+json")
                || (j.op == "search_custom"
                    && j.response_format == "text"
                    && ["application/xml", "text/xml", "text/plain", "text/csv"]
                        .contains(&ct.as_str()))
        };
        if !valid {
            return Err(Failure {
                code: "WEB_CONTENT_TYPE_DENIED",
                status: Some(status),
                trace,
            });
        }
        if response
            .headers()
            .get("content-encoding")
            .is_some_and(|v| v != "identity")
        {
            return Err(Failure {
                code: "WEB_CONTENT_ENCODING_DENIED",
                status: Some(status),
                trace,
            });
        }
        let mut bytes = Vec::new();
        response
            .by_ref()
            .take((limit + 1) as u64)
            .read_to_end(&mut bytes)
            .map_err(|_| Failure {
                code: if started.elapsed() >= Duration::from_millis(j.timeout_ms) {
                    "WEB_TIMEOUT"
                } else {
                    "WEB_RESPONSE_INCOMPLETE"
                },
                status: Some(status),
                trace: trace.clone(),
            })?;
        if bytes.len() > limit {
            return Err(Failure {
                code: "SEARCH_RESPONSE_TOO_LARGE",
                status: Some(status),
                trace,
            });
        }
        let hash = format!("{:x}", Sha256::digest(&bytes));
        if j.op == "fetch" {
            return Ok(
                json!({"url":u.as_str(),"text":extract(&bytes,&ct)?,"raw_sha256":hash,"network_receipt":trace,"evidence_kind":"FETCHED_PAGE"}),
            );
        }
        let obj: Value = if j.op == "search_custom" && j.response_format == "text" {
            Value::String(String::from_utf8(bytes).map_err(|_| fail("WEB_PROVIDER_INVALID"))?)
        } else {
            serde_json::from_slice(&bytes).map_err(|_| fail("WEB_PROVIDER_INVALID"))?
        };
        if j.op == "search_custom" {
            return Ok(
                json!({"payload":obj,"raw_sha256":hash,"network_receipt":trace,"evidence_kind":"SEARCH_SNIPPET"}),
            );
        }
        return Ok(
            json!({"rows":normalize(&j.provider,&obj)?,"raw_sha256":hash,"network_receipt":trace,"evidence_kind":"SEARCH_SNIPPET"}),
        );
    }
    if trace["attempts"]
        .as_array()
        .is_some_and(|a| a.iter().any(|v| v["result"] == "WEB_TIMEOUT"))
    {
        last.code = "WEB_TIMEOUT";
    }
    last.trace = trace;
    Err(last)
}
fn main() {
    let mut input = Vec::new();
    let result = io::stdin()
        .take(131073)
        .read_to_end(&mut input)
        .map_err(|_| fail("INVALID_SEARCH_PROVIDER"))
        .and_then(|_| {
            if input.len() > 131072 {
                return Err(fail("INVALID_SEARCH_PROVIDER"));
            }
            let j: Job =
                serde_json::from_slice(&input).map_err(|_| fail("INVALID_SEARCH_PROVIDER"))?;
            perform(j)
        });
    let value = match result {
        Ok(data) => json!({"ok":true,"data":data}),
        Err(e) => {
            json!({"ok":false,"error":{"code":e.code,"http_status":e.status,"network_receipt":e.trace}})
        }
    };
    println!("{value}");
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn block_private_and_tunnels() {
        for s in [
            "127.0.0.1",
            "10.0.0.1",
            "100.64.0.1",
            "169.254.169.254",
            "192.168.1.1",
            "192.0.2.1",
            "198.18.0.1",
            "224.1.1.1",
            "::1",
            "::ffff:8.8.8.8",
            "64:ff9b::808:808",
            "2001::abcd",
            "2002:808:808::1",
            "2001:db8::1",
        ] {
            assert!(!public_ip(s.parse().unwrap()), "{s}");
        }
    }
    #[test]
    fn allow_global() {
        for s in ["8.8.8.8", "104.18.10.203", "2606:4700:4700::1111"] {
            assert!(public_ip(s.parse().unwrap()));
        }
    }
    #[test]
    fn whole_dns_set_failclosed() {
        assert!(validate_addresses(vec![
            "8.8.8.8:443".parse().unwrap(),
            "127.0.0.1:443".parse().unwrap()
        ])
        .is_err());
    }
    #[test]
    fn url_security() {
        for s in [
            "http://example.org/",
            "https://127.0.0.1/",
            "https://a.local/",
            "https://a.internal/",
            "https://user:pw@example.org/",
            "https://example.org:444/",
            "https://example.org/?token=1234",
            "https://example.org/\n",
        ] {
            assert!(checked_url(s).is_err(), "{s}");
        }
    }
    #[test]
    fn snippet_is_not_page() {
        let rows=normalize("deepseek",&json!({"results":[{"url":"https://example.org/a","title":"A","summary":"lead","content":"ignored"}]})).unwrap();
        assert_eq!(rows[0]["evidence_kind"], "SEARCH_SNIPPET");
        assert_eq!(rows[0]["excerpt"], "lead");
    }
    #[test]
    fn malformed_is_not_empty() {
        assert!(normalize("bigmodel", &json!({"error":{"message":"secret"}})).is_err());
        assert_eq!(
            normalize("bigmodel", &json!({"search_result":[]})).unwrap(),
            json!([])
        );
    }
    #[test]
    fn html_is_not_executed() {
        let out=extract(b"<html><body><nav>menu</nav><h1>Article</h1><p>A &amp; B</p><script>attack()</script></body></html>","text/html").unwrap();
        assert!(out.contains("Article"));
        assert!(out.contains("A & B"));
        assert!(!out.contains("attack"));
        assert!(!out.contains("menu"));
    }
    #[test]
    fn provider_endpoints_fixed() {
        let j = Job {
            op: "search".into(),
            url: "https://evil.org/".into(),
            query: "hello".into(),
            provider: "deepseek".into(),
            key: "test".into(),
            timeout_ms: 12000,
            ..Default::default()
        };
        let (url, b) = request_spec(&j).unwrap();
        assert_eq!(url, "https://api.deepseek.com/v1/tools/search");
        assert_eq!(b.unwrap()["need_content"], false);
    }
    #[test]
    fn oversized_query_not_truncated() {
        let j = Job {
            op: "search".into(),
            url: String::new(),
            query: "中".repeat(129),
            provider: "deepseek".into(),
            key: "test".into(),
            timeout_ms: 12000,
            ..Default::default()
        };
        assert_eq!(request_spec(&j).unwrap_err().code, "INVALID_SEARCH_QUERY");
    }
    #[test]
    fn custom_endpoint_is_host_bound_with_encoded_credentials() {
        let j = Job {
            op: "search_custom".into(),
            url: "https://search.example.org/v2".into(),
            method: "GET".into(),
            body_format: "json".into(),
            response_format: "json".into(),
            params: BTreeMap::from([
                ("q".into(), json!("hello&api_key=attacker")),
                ("api_key".into(), json!("host-secret")),
            ]),
            ..Default::default()
        };
        let (url, _) = request_spec(&j).unwrap();
        let u = Url::parse(&url).unwrap();
        assert_eq!(u.host_str(), Some("search.example.org"));
        assert_eq!(u.path(), "/v2");
        assert_eq!(u.query_pairs().filter(|(k, _)| k == "api_key").count(), 1);
        assert_eq!(
            u.query_pairs().find(|(k, _)| k == "q").unwrap().1,
            "hello&api_key=attacker"
        );
    }
    #[test]
    fn custom_protocol_does_not_weaken_page_ssrf() {
        let mut j = Job {
            op: "search_custom".into(),
            url: "https://127.0.0.1/".into(),
            method: "GET".into(),
            body_format: "json".into(),
            response_format: "json".into(),
            ..Default::default()
        };
        assert_eq!(request_spec(&j).unwrap_err().code, "UNSAFE_SOURCE_URL");
        j.url = "https://search.example.org/".into();
        j.headers.insert("Host".into(), "169.254.169.254".into());
        assert_eq!(
            request_spec(&j).unwrap_err().code,
            "INVALID_SEARCH_PROVIDER"
        );
        j.headers.clear();
        j.headers
            .insert("X-Key".into(), "a\r\nHost: internal".into());
        assert!(request_spec(&j).is_err());
    }
    #[test]
    fn generic_method_and_form_shape_are_checked_before_dns() {
        let mut j = Job {
            op: "search_custom".into(),
            url: "https://search.example.org/".into(),
            method: "DELETE".into(),
            body_format: "json".into(),
            response_format: "json".into(),
            ..Default::default()
        };
        assert!(request_spec(&j).is_err());
        j.method = "POST".into();
        j.body_format = "form".into();
        j.body = Some(json!({"query":{"nested":true}}));
        assert!(request_spec(&j).is_err());
        j.body = Some(json!({"query":"hello","count":8}));
        assert!(request_spec(&j).is_ok());
    }
}
