//! A deliberately bounded RFC 8259 JSON codec for an offline, dependency-free wire layer.
//! Rejects duplicate object keys, invalid UTF-8, noncanonical number grammar, invalid
//! surrogates, trailing data, unsafe depth and excessive structure. Not a JSON Schema engine.
use crate::errors::{err, ErrorCode, Result};
use std::collections::BTreeMap;

pub const MAX_JSON_DEPTH: usize = 32;
pub const MAX_JSON_NODES: usize = 16384;
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Value {
    Null,
    Bool(bool),
    Number(String),
    String(String),
    Array(Vec<Value>),
    Object(BTreeMap<String, Value>),
}
impl Value {
    pub fn get(&self, name: &str) -> Option<&Self> {
        match self {
            Self::Object(map) => map.get(name),
            _ => None,
        }
    }
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::String(v) => Some(v),
            _ => None,
        }
    }
    pub fn as_u64(&self) -> Option<u64> {
        match self {
            Self::Number(n) if !n.starts_with('-') && n.bytes().all(|c| c.is_ascii_digit()) => {
                n.parse().ok()
            }
            _ => None,
        }
    }
    pub fn object(items: impl IntoIterator<Item = (String, Value)>) -> Self {
        Self::Object(items.into_iter().collect())
    }
    pub fn encode(&self) -> String {
        let mut out = String::new();
        self.write_into(&mut out);
        out
    }
    fn write_into(&self, out: &mut String) {
        match self {
            Self::Null => out.push_str("null"),
            Self::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
            Self::Number(n) => out.push_str(n),
            Self::String(s) => write_string(out, s),
            Self::Array(a) => {
                out.push('[');
                for (i, value) in a.iter().enumerate() {
                    if i != 0 {
                        out.push(',');
                    }
                    value.write_into(out);
                }
                out.push(']');
            }
            Self::Object(map) => {
                out.push('{');
                for (i, (key, value)) in map.iter().enumerate() {
                    if i != 0 {
                        out.push(',');
                    }
                    write_string(out, key);
                    out.push(':');
                    value.write_into(out);
                }
                out.push('}');
            }
        }
    }
    pub fn depth_and_nodes(&self) -> Option<usize> {
        fn check(v: &Value, depth: usize, nodes: &mut usize) -> Option<()> {
            *nodes = nodes.checked_add(1)?;
            if *nodes > MAX_JSON_NODES || depth > MAX_JSON_DEPTH {
                return None;
            }
            match v {
                Value::Array(a) => {
                    for child in a {
                        check(child, depth + 1, nodes)?;
                    }
                }
                Value::Object(o) => {
                    for child in o.values() {
                        check(child, depth + 1, nodes)?;
                    }
                }
                _ => {}
            }
            Some(())
        }
        let mut nodes = 0;
        check(self, 0, &mut nodes)?;
        Some(nodes)
    }
}
fn write_string(out: &mut String, value: &str) {
    out.push('"');
    for c in value.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{08}' => out.push_str("\\b"),
            '\u{0c}' => out.push_str("\\f"),
            c if c < '\u{20}' => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
}

pub fn parse(bytes: &[u8]) -> Result<Value> {
    let text =
        std::str::from_utf8(bytes).map_err(|_| err(ErrorCode::InvalidJson, "invalid UTF-8"))?;
    let mut parser = Parser {
        text,
        pos: 0,
        nodes: 0,
    };
    parser.ws();
    let value = parser.value(0)?;
    parser.ws();
    if parser.pos != text.len() {
        return Err(err(ErrorCode::InvalidJson, "trailing JSON data"));
    }
    Ok(value)
}
struct Parser<'a> {
    text: &'a str,
    pos: usize,
    nodes: usize,
}
impl Parser<'_> {
    fn ws(&mut self) {
        while matches!(self.peek(), Some(b' ' | b'\r' | b'\n' | b'\t')) {
            self.pos += 1;
        }
    }
    fn peek(&self) -> Option<u8> {
        self.text.as_bytes().get(self.pos).copied()
    }
    fn eat(&mut self, c: u8) -> bool {
        if self.peek() == Some(c) {
            self.pos += 1;
            true
        } else {
            false
        }
    }
    fn expect(&mut self, c: u8) -> Result<()> {
        if self.eat(c) {
            Ok(())
        } else {
            Err(err(ErrorCode::InvalidJson, "JSON punctuation error"))
        }
    }
    fn value(&mut self, depth: usize) -> Result<Value> {
        self.nodes += 1;
        if depth > MAX_JSON_DEPTH || self.nodes > MAX_JSON_NODES {
            return Err(err(
                ErrorCode::InvalidJson,
                "JSON complexity limit exceeded",
            ));
        }
        match self.peek() {
            Some(b'{') => self.object(depth),
            Some(b'[') => self.array(depth),
            Some(b'"') => Ok(Value::String(self.string()?)),
            Some(b't') => {
                self.keyword("true")?;
                Ok(Value::Bool(true))
            }
            Some(b'f') => {
                self.keyword("false")?;
                Ok(Value::Bool(false))
            }
            Some(b'n') => {
                self.keyword("null")?;
                Ok(Value::Null)
            }
            Some(b'-' | b'0'..=b'9') => Ok(Value::Number(self.number()?)),
            _ => Err(err(ErrorCode::InvalidJson, "invalid JSON value")),
        }
    }
    fn keyword(&mut self, keyword: &str) -> Result<()> {
        if self.text[self.pos..].starts_with(keyword) {
            self.pos += keyword.len();
            Ok(())
        } else {
            Err(err(ErrorCode::InvalidJson, "invalid JSON literal"))
        }
    }
    fn object(&mut self, depth: usize) -> Result<Value> {
        self.expect(b'{')?;
        self.ws();
        let mut map = BTreeMap::new();
        if self.eat(b'}') {
            return Ok(Value::Object(map));
        }
        loop {
            if self.peek() != Some(b'"') {
                return Err(err(
                    ErrorCode::InvalidJson,
                    "JSON object key must be string",
                ));
            }
            let key = self.string()?;
            self.ws();
            self.expect(b':')?;
            self.ws();
            let v = self.value(depth + 1)?;
            if map.insert(key, v).is_some() {
                return Err(err(ErrorCode::InvalidJson, "duplicate JSON object key"));
            }
            self.ws();
            if self.eat(b'}') {
                break;
            }
            self.expect(b',')?;
            self.ws();
        }
        Ok(Value::Object(map))
    }
    fn array(&mut self, depth: usize) -> Result<Value> {
        self.expect(b'[')?;
        self.ws();
        let mut out = Vec::new();
        if self.eat(b']') {
            return Ok(Value::Array(out));
        }
        loop {
            out.push(self.value(depth + 1)?);
            self.ws();
            if self.eat(b']') {
                break;
            }
            self.expect(b',')?;
            self.ws();
        }
        Ok(Value::Array(out))
    }
    fn hex4(&mut self) -> Result<u16> {
        let end = self
            .pos
            .checked_add(4)
            .ok_or_else(|| err(ErrorCode::InvalidJson, "invalid escape"))?;
        let h = self
            .text
            .get(self.pos..end)
            .ok_or_else(|| err(ErrorCode::InvalidJson, "short Unicode escape"))?;
        let num = u16::from_str_radix(h, 16)
            .map_err(|_| err(ErrorCode::InvalidJson, "invalid Unicode escape"))?;
        self.pos = end;
        Ok(num)
    }
    fn string(&mut self) -> Result<String> {
        self.expect(b'"')?;
        let mut result = String::new();
        let mut segment = self.pos;
        loop {
            let byte = self
                .peek()
                .ok_or_else(|| err(ErrorCode::InvalidJson, "unterminated JSON string"))?;
            match byte {
                b'"' => {
                    result.push_str(&self.text[segment..self.pos]);
                    self.pos += 1;
                    return Ok(result);
                }
                b'\\' => {
                    result.push_str(&self.text[segment..self.pos]);
                    self.pos += 1;
                    let code = self
                        .peek()
                        .ok_or_else(|| err(ErrorCode::InvalidJson, "unfinished escape"))?;
                    self.pos += 1;
                    match code {
                        b'"' => result.push('"'),
                        b'\\' => result.push('\\'),
                        b'/' => result.push('/'),
                        b'b' => result.push('\u{08}'),
                        b'f' => result.push('\u{0c}'),
                        b'n' => result.push('\n'),
                        b'r' => result.push('\r'),
                        b't' => result.push('\t'),
                        b'u' => {
                            let hi = self.hex4()?;
                            let scalar = if (0xd800..=0xdbff).contains(&hi) {
                                self.expect(b'\\')?;
                                self.expect(b'u')?;
                                let lo = self.hex4()?;
                                if !(0xdc00..=0xdfff).contains(&lo) {
                                    return Err(err(
                                        ErrorCode::InvalidJson,
                                        "invalid surrogate pair",
                                    ));
                                }
                                0x10000 + ((u32::from(hi) - 0xd800) << 10) + u32::from(lo) - 0xdc00
                            } else if (0xdc00..=0xdfff).contains(&hi) {
                                return Err(err(ErrorCode::InvalidJson, "unpaired surrogate"));
                            } else {
                                u32::from(hi)
                            };
                            result.push(char::from_u32(scalar).ok_or_else(|| {
                                err(ErrorCode::InvalidJson, "invalid Unicode scalar")
                            })?);
                        }
                        _ => return Err(err(ErrorCode::InvalidJson, "invalid JSON escape")),
                    }
                    segment = self.pos;
                }
                0..=31 => {
                    return Err(err(
                        ErrorCode::InvalidJson,
                        "unescaped control in JSON string",
                    ))
                }
                _ => {
                    let c = self.text[self.pos..]
                        .chars()
                        .next()
                        .ok_or_else(|| err(ErrorCode::InvalidJson, "bad string UTF-8"))?;
                    self.pos += c.len_utf8();
                }
            }
        }
    }
    fn number(&mut self) -> Result<String> {
        let start = self.pos;
        self.eat(b'-');
        if !self.eat(b'0') {
            if !matches!(self.peek(), Some(b'1'..=b'9')) {
                return Err(err(ErrorCode::InvalidJson, "invalid JSON integer"));
            }
            while matches!(self.peek(), Some(b'0'..=b'9')) {
                self.pos += 1;
            }
        }
        if self.eat(b'.') {
            if !matches!(self.peek(), Some(b'0'..=b'9')) {
                return Err(err(ErrorCode::InvalidJson, "invalid JSON fraction"));
            }
            while matches!(self.peek(), Some(b'0'..=b'9')) {
                self.pos += 1;
            }
        }
        if self.eat(b'e') || self.eat(b'E') {
            if !self.eat(b'+') {
                self.eat(b'-');
            }
            if !matches!(self.peek(), Some(b'0'..=b'9')) {
                return Err(err(ErrorCode::InvalidJson, "invalid JSON exponent"));
            }
            while matches!(self.peek(), Some(b'0'..=b'9')) {
                self.pos += 1;
            }
        }
        Ok(self.text[start..self.pos].to_string())
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn unicode_roundtrip() {
        let s = r#"{"emoji":"\uD83E\uDDEA","chinese":"验证","tab":"x\t","quote":"\""}"#;
        let value = parse(s.as_bytes()).unwrap();
        assert_eq!(parse(value.encode().as_bytes()).unwrap(), value);
    }
    #[test]
    fn reject_invalid_json() {
        let cases = [
            r#"{"a":1,"a":2}"#,
            "[1,]",
            "{\"a\":1,}",
            "1e+",
            "01",
            "--2",
            "[[]",
            "\"\\ud800\"",
            "\"\\ud800\\u0061\"",
            "\"\\udc00\"",
            "\"abc\n\"",
            "null false",
            "{\"x\":true",
        ];
        for c in cases {
            assert!(parse(c.as_bytes()).is_err(), "accepted: {c}");
        }
    }
    #[test]
    fn complexity_limits() {
        assert!(parse(format!("{}0{}", "[".repeat(34), "]".repeat(34)).as_bytes()).is_err());
        assert!(parse(format!("[{}]", "0,".repeat(18000) + "0").as_bytes()).is_err());
    }
    #[test]
    fn numeric_fidelity() {
        let v = parse(b"9223372036854775808").unwrap();
        assert_eq!(v.encode(), "9223372036854775808");
        assert!(v.as_u64().is_some());
        assert_eq!(parse(b"-1").unwrap().as_u64(), None);
    }
}
