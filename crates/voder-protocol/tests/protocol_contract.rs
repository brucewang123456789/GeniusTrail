use voder_protocol::{
    commands::{Command, ConnectionAdmission},
    envelope::{Envelope, FrameKind, MAX_FRAME_BYTES, PROTOCOL_VERSION},
    errors::{ErrorCode, ProtocolError},
    events::{EventCursor, EventType, ServerEvent},
    json::{self, Value},
};

fn obj(items: impl IntoIterator<Item = (&'static str, Value)>) -> Value {
    Value::object(items.into_iter().map(|(k, v)| (k.to_string(), v)))
}
fn s(value: &str) -> Value {
    Value::String(value.into())
}
fn n(value: u64) -> Value {
    Value::Number(value.to_string())
}
fn hello() -> Envelope {
    Envelope::request(
        "req-1",
        "initialize",
        obj([
            ("client_name", s("Voder-CLI")),
            ("minimum_version", n(1)),
            ("maximum_version", n(1)),
            (
                "requested_features",
                Value::Array(vec![s("jsonl"), s("event-resume")]),
            ),
        ]),
    )
}
#[test]
fn server_response_and_error_correlation() {
    let request = hello();
    let reply = Envelope::respond_to(&request, obj([("accepted", Value::Bool(true))])).unwrap();
    assert_eq!(reply.request_id, request.request_id);
    assert_eq!(reply.frame_type, FrameKind::Response);
    assert_eq!(
        Envelope::decode(&reply.encode_line().unwrap()).unwrap(),
        reply
    );
    let reject = Envelope::reject(
        &request,
        &ProtocolError::new(ErrorCode::Unauthorized, "approval required"),
    )
    .unwrap();
    assert_eq!(reject.frame_type, FrameKind::Error);
    assert_eq!(reject.payload.get("code"), Some(&s("UNAUTHORIZED")));
    assert_eq!(reject.request_id, request.request_id);
}
#[test]
fn session_create_before_session_exists() {
    let mut conn = ConnectionAdmission::default();
    conn.admit(&hello()).unwrap();
    let mut create = Envelope::request(
        "req-2",
        "session.create",
        obj([("project_root", s("/work/repo"))]),
    );
    create.idempotency_key = Some("create-1".into());
    assert!(matches!(
        conn.admit(&create).unwrap(),
        Command::SessionCreate { .. }
    ));
    create.request_id = Some("req-3".into());
    create.session_id = Some("forged-session".into());
    assert_eq!(
        conn.admit(&create).unwrap_err().code,
        ErrorCode::InvalidField
    );
}
#[test]
fn model_cannot_turn_protocol_into_authority() {
    let methods = [
        "state.set",
        "mark_done",
        "force_evidence_pass",
        "grant.approval",
        "tool.request",
        "action.commit",
        "evidence.admitted",
        "validation.passed",
        "sandbox.disable",
        "memory.promote",
        "budget.override",
        "worktree.force_delete",
    ];
    for (i, method) in methods.iter().enumerate() {
        let mut request = Envelope::request(format!("r{i}"), *method, obj([]));
        request.session_id = Some("s-1".into());
        assert_eq!(
            Command::parse(&request).unwrap_err().code,
            ErrorCode::UnsupportedMethod
        );
    }
}
#[test]
fn context_reference_is_hash_only_not_mutable_memory() {
    let mut request = Envelope::request(
        "r",
        "task.submit",
        obj([
            ("instruction", s("Fix the tests")),
            (
                "context_refs",
                Value::Array(vec![s(&format!("sha256:{}", "a".repeat(64)))]),
            ),
        ]),
    );
    request.session_id = Some("session-1".into());
    request.idempotency_key = Some("submit-1".into());
    assert!(matches!(
        Command::parse(&request).unwrap(),
        Command::TaskSubmit { .. }
    ));
    if let Value::Object(ref mut map) = request.payload {
        map.insert(
            "context_refs".into(),
            Value::Array(vec![s("/root/secrets")]),
        );
    }
    assert_eq!(
        Command::parse(&request).unwrap_err().code,
        ErrorCode::InvalidPayload
    );
}
#[test]
fn malformed_payload_field_rejected_fail_closed() {
    let mut req = Envelope::request("r", "task.cancel", obj([("run_id", s("run-1"))]));
    req.session_id = Some("session".into());
    req.idempotency_key = Some("cancel".into());
    if let Value::Object(ref mut map) = req.payload {
        map.insert("force".into(), Value::Bool(true));
    }
    assert_eq!(
        Command::parse(&req).unwrap_err().code,
        ErrorCode::InvalidPayload
    );
}
#[test]
fn wire_identifiers_reject_control_escape_and_confusables() {
    for id in [
        "../secret",
        "x y",
        "甲",
        "a\0b",
        "\n",
        "x/y",
        "x\\y",
        "",
        "a".repeat(129).as_str(),
    ] {
        let req = Envelope::request(id, "initialize", obj([]));
        assert!(req.encode_line().is_err());
    }
}
#[test]
fn malformed_outbound_json_number_not_emitted() {
    for bad in ["01", "NaN", "Infinity", "-", "1.", "1e", "--3"] {
        let req = Envelope::request(
            "r",
            "initialize",
            obj([("bad", Value::Number(bad.to_string()))]),
        );
        assert!(req.encode_line().is_err(), "encoded invalid number {bad}");
    }
}
#[test]
fn event_sequence_and_resume_exactly_monotone_in_memory() {
    let mut cursor = EventCursor::default();
    for seq in 1..=24 {
        let event = ServerEvent {
            kind: EventType::ToolCompleted,
            session_id: "s".into(),
            sequence: seq,
            payload: obj([]),
        };
        cursor.accept(&event).unwrap();
    }
    assert!(cursor
        .accept(&ServerEvent {
            kind: EventType::RunCompleted,
            session_id: "s".into(),
            sequence: 26,
            payload: obj([])
        })
        .is_err());
    cursor.start_at("s", 25);
    cursor
        .accept(&ServerEvent {
            kind: EventType::RunCompleted,
            session_id: "s".into(),
            sequence: 26,
            payload: obj([]),
        })
        .unwrap();
}
#[test]
fn contract_manifest_matches_implemented_methods_and_events() {
    let manifest = voder_protocol::contract_manifest();
    let parsed = json::parse(manifest.encode().as_bytes()).unwrap();
    assert_eq!(parsed.get("protocol_version"), Some(&n(PROTOCOL_VERSION)));
    let Value::Array(cmds) = parsed.get("commands").unwrap() else {
        panic!("commands list missing");
    };
    for c in cmds {
        assert!(voder_protocol::commands::CLIENT_METHODS.contains(&c.as_str().unwrap()));
    }
    assert_eq!(cmds.len(), 10);
    let Value::Array(events) = parsed.get("events").unwrap() else {
        panic!("events missing");
    };
    assert_eq!(events.len(), 14);
}
#[test]
fn oversize_frames_and_jsonl_smuggling_rejected() {
    let msg = hello().encode_line().unwrap();
    let mut appended = msg.clone();
    appended.extend_from_slice(&msg);
    assert!(Envelope::decode(&appended).is_err());
    assert_eq!(
        Envelope::decode(&vec![b' '; MAX_FRAME_BYTES + 1])
            .unwrap_err()
            .code,
        ErrorCode::FrameTooLarge
    );
}
#[test]
fn json_fuzz_deterministic_no_panics() {
    let dictionary = b"{}[],:.\\\"abABC0011eE-+truefalse null\t\r\n\x00\x7f\xff\xc3\xa9";
    let mut rng = 0x00de_adbe_efca_fe15_u64;
    for _ in 0..15000 {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        let len = (rng as usize % 256) + 1;
        let mut data = Vec::with_capacity(len);
        for _ in 0..len {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            data.push(dictionary[(rng as usize) % dictionary.len()]);
        }
        // The parser is also the adversarial input boundary for all wire messages.
        let parsed = std::panic::catch_unwind(|| json::parse(&data));
        assert!(parsed.is_ok(), "JSON decoder panicked");
        if let Ok(value) = parsed.unwrap() {
            let encoded = value.encode();
            assert_eq!(json::parse(encoded.as_bytes()).unwrap(), value);
        }
        assert!(std::panic::catch_unwind(|| Envelope::decode(&data)).is_ok());
    }
}
