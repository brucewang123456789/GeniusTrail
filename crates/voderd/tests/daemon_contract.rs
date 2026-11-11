use std::{
    fs,
    io::Write,
    path::{Path, PathBuf},
    process::{Command, Stdio},
    sync::atomic::{AtomicU64, Ordering},
};
use voder_protocol::{
    envelope::{Envelope, FrameKind},
    json::Value,
};
use voderd::{
    journal::{Journal, ZERO_HASH},
    protocol_stdio::Server,
    session::State,
};
static SEQ: AtomicU64 = AtomicU64::new(0);
struct Fixture {
    home: PathBuf,
    repo: PathBuf,
    state: PathBuf,
}
impl Fixture {
    fn new() -> Self {
        let id = SEQ.fetch_add(1, Ordering::Relaxed);
        let home = std::env::temp_dir().join(format!(
            "voderd-test-{}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos(),
            id
        ));
        fs::create_dir_all(&home).unwrap();
        let repo = home.join("repo");
        fs::create_dir(&repo).unwrap();
        assert!(Command::new("git")
            .arg("init")
            .arg("-q")
            .arg(&repo)
            .status()
            .unwrap()
            .success());
        fs::write(repo.join("README.md"), "voder test repository\n").unwrap();
        assert!(Command::new("git")
            .args(["-C", repo.to_str().unwrap(), "add", "README.md"])
            .status()
            .unwrap()
            .success());
        assert!(Command::new("git")
            .args([
                "-C",
                repo.to_str().unwrap(),
                "-c",
                "user.name=Voder Test",
                "-c",
                "user.email=voder@example.invalid",
                "commit",
                "-qm",
                "initial"
            ])
            .status()
            .unwrap()
            .success());
        let state = home.join("state");
        Self { home, repo, state }
    }
    fn start(&self) -> Server {
        Server::open(&self.state).unwrap()
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.home);
    }
}
fn req(id: &str, method: &str, session: Option<&str>, key: Option<&str>, props: Value) -> Envelope {
    let mut r = Envelope::request(id, method, props);
    r.session_id = session.map(str::to_owned);
    r.idempotency_key = key.map(str::to_owned);
    r
}
fn props(k: &str, v: Value) -> Value {
    Value::object([(k.into(), v)])
}
fn s(v: &str) -> Value {
    Value::String(v.into())
}
fn init(srv: &mut Server) {
    let mut r = req(
        "hello",
        "initialize",
        None,
        None,
        Value::object([
            ("minimum_version".into(), Value::Number("1".into())),
            ("maximum_version".into(), Value::Number("1".into())),
            ("client_name".into(), s("voder-test")),
            ("requested_features".into(), Value::Array(vec![])),
        ]),
    );
    assert_eq!(
        srv.handle(r.clone()).unwrap().frame_type,
        FrameKind::Response
    );
    r.request_id = Some("hello-again".into());
    assert_eq!(srv.handle(r).unwrap().frame_type, FrameKind::Error);
}
fn create(srv: &mut Server, repo: &Path, key: &str, id: &str) -> String {
    let r = req(
        id,
        "session.create",
        None,
        Some(key),
        props("project_root", s(repo.to_str().unwrap())),
    );
    let result = srv.handle(r).unwrap();
    assert_eq!(result.frame_type, FrameKind::Response, "{result:?}");
    result
        .payload
        .get("session_id")
        .unwrap()
        .as_str()
        .unwrap()
        .into()
}
fn submit(srv: &mut Server, session: &str, id: &str, key: &str, context: Vec<Value>) -> Envelope {
    let r = req(
        id,
        "task.submit",
        Some(session),
        Some(key),
        Value::object([
            ("instruction".into(), s("fix broken parser")),
            ("context_refs".into(), Value::Array(context)),
        ]),
    );
    srv.handle(r).unwrap()
}
#[test]
fn session_create_persisted_and_idempotent_after_restart() {
    let f = Fixture::new();
    let mut a = f.start();
    init(&mut a);
    let first = create(&mut a, &f.repo, "key-one", "create-1");
    assert_eq!(a.event_count(), 1);
    assert_eq!(create(&mut a, &f.repo, "key-one", "create-2"), first);
    assert_eq!(a.event_count(), 1);
    drop(a);
    let mut b = f.start();
    init(&mut b);
    assert_eq!(create(&mut b, &f.repo, "key-one", "create-3"), first);
    assert_eq!(
        b.state().get(&first).unwrap().project_root,
        f.repo.to_str().unwrap()
    );
    assert_eq!(b.event_count(), 1);
    assert!(b.is_locked());
}
#[test]
fn different_request_under_same_idempotency_key_conflicts() {
    let f = Fixture::new();
    let mut a = f.start();
    init(&mut a);
    create(&mut a, &f.repo, "k", "c1");
    let r = req(
        "c2",
        "session.create",
        None,
        Some("k"),
        props("project_root", s("/tmp/different")),
    );
    let e = a.handle(r).unwrap();
    assert_eq!(e.frame_type, FrameKind::Error);
    assert_eq!(e.payload.get("code").unwrap().as_str(), Some("CONFLICT"));
    assert_eq!(a.event_count(), 1);
}
#[test]
fn concurrent_writer_rejected_then_unlocks() {
    let f = Fixture::new();
    let a = f.start();
    assert!(Server::open(&f.state).is_err());
    drop(a);
    assert!(Server::open(&f.state).is_ok());
}
#[test]
fn missing_context_blocks_no_fake_delivery_and_recovers() {
    let f = Fixture::new();
    let mut a = f.start();
    init(&mut a);
    let sid = create(&mut a, &f.repo, "k1", "c1");
    let missing = format!("sha256:{}", ZERO_HASH);
    let out = submit(&mut a, &sid, "s1", "k2", vec![s(&missing)]);
    assert_eq!(out.payload.get("status").unwrap().as_str(), Some("BLOCKED"));
    assert!(out
        .payload
        .get("reason")
        .unwrap()
        .as_str()
        .unwrap()
        .contains("MISSING_CONTEXT"));
    assert_eq!(
        out.payload.get("delivery_allowed"),
        Some(&Value::Bool(false))
    );
    let run = out
        .payload
        .get("run_id")
        .unwrap()
        .as_str()
        .unwrap()
        .to_owned();
    assert_eq!(a.event_count(), 2);
    drop(a);
    let mut b = f.start();
    init(&mut b);
    assert_eq!(
        submit(&mut b, &sid, "s2", "k2", vec![s(&missing)]).payload,
        out.payload
    );
    assert_eq!(b.event_count(), 2);
    let read = req(
        "state1",
        "state.get",
        Some(&sid),
        None,
        props("run_id", s(&run)),
    );
    let value = b.handle(read).unwrap().payload;
    assert_eq!(
        value.get("runs").unwrap().clone(),
        Value::Array(vec![Value::object([
            ("run_id".into(), s(&run)),
            ("status".into(), s("BLOCKED")),
            ("delivery_allowed".into(), Value::Bool(false)),
            (
                "reason".into(),
                s("MISSING_CONTEXT: supply referenced context artifact; no tool execution")
            )
        ])])
    );
}
#[test]
fn no_provider_blocks_clean_task() {
    let f = Fixture::new();
    let mut a = f.start();
    init(&mut a);
    let sid = create(&mut a, &f.repo, "a", "c");
    let result = submit(&mut a, &sid, "s", "b", vec![]);
    assert!(result
        .payload
        .get("reason")
        .unwrap()
        .as_str()
        .unwrap()
        .contains("PROVIDER_UNAVAILABLE"));
}
#[test]
fn cancel_and_replay_have_real_sequence() {
    let f = Fixture::new();
    let mut a = f.start();
    init(&mut a);
    let sid = create(&mut a, &f.repo, "a", "c");
    let out = submit(&mut a, &sid, "s", "b", vec![]);
    let id = out.payload.get("run_id").unwrap().as_str().unwrap();
    let c = req(
        "cancel",
        "task.cancel",
        Some(&sid),
        Some("z"),
        props("run_id", s(id)),
    );
    assert_eq!(
        a.handle(c).unwrap().payload.get("status").unwrap().as_str(),
        Some("CANCELLED")
    );
    let r = req(
        "replay",
        "run.replay",
        Some(&sid),
        None,
        Value::object([
            ("run_id".into(), s(id)),
            ("after_sequence".into(), Value::Number("0".into())),
            ("limit".into(), Value::Number("10".into())),
        ]),
    );
    let ev = a.handle(r).unwrap();
    let Value::Array(events) = ev.payload.get("events").unwrap() else {
        panic!()
    };
    assert_eq!(events.len(), 2);
    assert_eq!(a.event_count(), 3);
}
#[test]
fn no_unauthorized_tools_state_override_or_approval() {
    let f = Fixture::new();
    let mut a = f.start();
    init(&mut a);
    let sid = create(&mut a, &f.repo, "a", "c");
    for (i, m) in [
        "state.set",
        "force_evidence_pass",
        "tool.execute",
        "mark_done",
    ]
    .iter()
    .enumerate()
    {
        let e = a
            .handle(req(
                &format!("a{i}"),
                m,
                Some(&sid),
                None,
                Value::object([]),
            ))
            .unwrap();
        assert_eq!(e.frame_type, FrameKind::Error);
    }
    let e = a
        .handle(req(
            "approval",
            "approval.respond",
            Some(&sid),
            Some("p"),
            Value::object([
                ("approval_id".into(), s("id")),
                ("decision".into(), s("approve")),
            ]),
        ))
        .unwrap();
    assert_eq!(e.frame_type, FrameKind::Error);
    assert_eq!(a.event_count(), 1);
}
#[test]
fn artifacts_are_not_cross_session_readable() {
    let f = Fixture::new();
    let mut a = f.start();
    init(&mut a);
    let sid = create(&mut a, &f.repo, "a", "c");
    let v = a
        .handle(req(
            "art",
            "artifact.read",
            Some(&sid),
            None,
            Value::object([
                ("digest".into(), s(&format!("sha256:{}", ZERO_HASH))),
                ("offset".into(), Value::Number("0".into())),
                ("limit".into(), Value::Number("10".into())),
            ]),
        ))
        .unwrap();
    assert_eq!(v.frame_type, FrameKind::Error);
    assert_eq!(
        v.payload.get("code").unwrap().as_str(),
        Some("UNAUTHORIZED")
    );
}
#[test]
fn journal_bitflip_and_torn_tail_rejected() {
    let f = Fixture::new();
    let mut s = f.start();
    init(&mut s);
    create(&mut s, &f.repo, "one", "c");
    drop(s);
    let p = f.state.join("events.jsonl");
    let original = fs::read(&p).unwrap();
    let mut changed = original.clone();
    let at = changed.iter().position(|c| *c == b's').unwrap();
    changed[at] = b'q';
    fs::write(&p, &changed).unwrap();
    assert!(Server::open(&f.state).is_err());
    fs::write(&p, &original).unwrap();
    let mut file = fs::OpenOptions::new().append(true).open(&p).unwrap();
    file.write_all(b"{\"torn\":").unwrap();
    drop(file);
    assert!(Server::open(&f.state).is_err());
}
#[cfg(unix)]
#[test]
fn journal_symlink_refused() {
    let f = Fixture::new();
    fs::create_dir(&f.state).unwrap();
    fs::write(f.home.join("outside"), b"x").unwrap();
    std::os::unix::fs::symlink(f.home.join("outside"), f.state.join("events.jsonl")).unwrap();
    assert!(Server::open(&f.state).is_err());
}
#[test]
fn journal_rejects_fabricated_authority() {
    let f = Fixture::new();
    let mut j = Journal::open(&{
        fs::create_dir(&f.state).unwrap();
        f.state.join("events.jsonl")
    })
    .unwrap();
    let bogus = Value::object([
        ("type".into(), s("force_evidence_pass")),
        ("session".into(), s("any")),
        ("key".into(), s("x")),
        ("fingerprint".into(), s("abc")),
        ("response".into(), Value::object([])),
    ]);
    let e = j.append(bogus).unwrap();
    assert!(State::reconstruct(&[e]).is_err());
    drop(j);
    assert!(Server::open(&f.state).is_err());
}
#[test]
fn malformed_frames_do_not_execute_actions() {
    let f = Fixture::new();
    let mut server = f.start();
    let err = server.serve(&b"not-json\n"[..], Vec::new()).unwrap_err();
    assert!(format!("{err}").contains("malformed"));
    assert_eq!(server.event_count(), 0);
}
#[test]
fn real_stdio_process_restart_and_replay() {
    let f = Fixture::new();
    let exe = env!("CARGO_BIN_EXE_voderd");
    let first = Envelope::request(
        "initialize-1",
        "initialize",
        Value::object([
            ("minimum_version".into(), Value::Number("1".into())),
            ("maximum_version".into(), Value::Number("1".into())),
            ("client_name".into(), s("cli")),
            ("requested_features".into(), Value::Array(vec![])),
        ]),
    );
    let create = req(
        "create-1",
        "session.create",
        None,
        Some("install"),
        props("project_root", s(f.repo.to_str().unwrap())),
    );
    let mut bytes = first.encode_line().unwrap();
    bytes.extend(create.encode_line().unwrap());
    let run = |input: &[u8]| {
        let mut child = Command::new(exe)
            .arg("--state-dir")
            .arg(&f.state)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        child.stdin.take().unwrap().write_all(input).unwrap();
        let out = child.wait_with_output().unwrap();
        assert!(
            out.status.success(),
            "{:?}",
            String::from_utf8_lossy(&out.stderr)
        );
        String::from_utf8(out.stdout).unwrap()
    };
    let out = run(&bytes);
    let lines: Vec<&str> = out.lines().collect();
    assert_eq!(lines.len(), 2);
    let session = Envelope::decode(lines[1].as_bytes())
        .unwrap()
        .payload
        .get("session_id")
        .unwrap()
        .as_str()
        .unwrap()
        .to_owned();
    let mut again = first.clone();
    again.request_id = Some("initialize-2".into());
    let mut create2 = create.clone();
    create2.request_id = Some("create-2".into());
    let mut second = again.encode_line().unwrap();
    second.extend(create2.encode_line().unwrap());
    let lines: Vec<String> = run(&second).lines().map(str::to_owned).collect();
    assert_eq!(
        Envelope::decode(lines[1].as_bytes())
            .unwrap()
            .payload
            .get("session_id")
            .unwrap()
            .as_str(),
        Some(session.as_str())
    );
    assert_eq!(
        fs::read_to_string(f.state.join("events.jsonl"))
            .unwrap()
            .lines()
            .count(),
        1
    );
}
#[test]
fn initial_state_requires_explicit_negotiate() {
    let f = Fixture::new();
    let mut srv = f.start();
    let e = srv
        .handle(req(
            "create",
            "session.create",
            None,
            Some("k"),
            props("project_root", s(f.repo.to_str().unwrap())),
        ))
        .unwrap();
    assert_eq!(e.frame_type, FrameKind::Error);
    assert_eq!(srv.event_count(), 0);
}

#[test]
fn one_session_id_cannot_read_another_run() {
    let f = Fixture::new();
    let mut d = f.start();
    init(&mut d);
    let s1 = create(&mut d, &f.repo, "make-one", "m1");
    let s2 = create(&mut d, &f.repo, "make-two", "m2");
    let r = submit(&mut d, &s1, "task", "job-1", vec![]);
    let run = r.payload.get("run_id").unwrap().as_str().unwrap();
    let other = d
        .handle(req(
            "other",
            "run.replay",
            Some(&s2),
            None,
            Value::object([
                ("run_id".into(), s(run)),
                ("after_sequence".into(), Value::Number("0".into())),
                ("limit".into(), Value::Number("10".into())),
            ]),
        ))
        .unwrap();
    assert_eq!(other.frame_type, FrameKind::Error);
}

#[test]
fn repeated_idempotency_keys_are_scoped_to_session() {
    let f = Fixture::new();
    let mut d = f.start();
    init(&mut d);
    let s1 = create(&mut d, &f.repo, "one", "c1");
    let s2 = create(&mut d, &f.repo, "two", "c2");
    assert_eq!(
        submit(&mut d, &s1, "a", "shared", vec![]).frame_type,
        FrameKind::Response
    );
    assert_eq!(
        submit(&mut d, &s2, "b", "shared", vec![]).frame_type,
        FrameKind::Response
    );
    assert_eq!(d.event_count(), 4);
}

#[test]
fn incomplete_or_oversized_frames_fail_closed() {
    let f = Fixture::new();
    let mut d = f.start();
    assert!(d.serve(&b"{\"truncated\":true"[..], Vec::new()).is_err());
    assert!(d.serve(&[b'x'; 1048577][..], Vec::new()).is_err());
    assert_eq!(d.event_count(), 0);
}

#[test]
fn no_authoritative_data_mutation_on_protocol_error() {
    let f = Fixture::new();
    let mut d = f.start();
    init(&mut d);
    let count = d.event_count();
    for (i, method) in ["task.submit", "task.cancel", "run.export", "session.resume"]
        .iter()
        .enumerate()
    {
        let mut r = req(
            &format!("bad{i}"),
            method,
            Some("fake"),
            Some("key"),
            Value::object([]),
        );
        if *method == "session.resume" {
            r.idempotency_key = None;
        }
        assert_eq!(d.handle(r).unwrap().frame_type, FrameKind::Error);
    }
    assert_eq!(d.event_count(), count);
}

#[test]
fn journal_rejects_oversized_record_at_startup() {
    let f = Fixture::new();
    fs::create_dir(&f.state).unwrap();
    fs::write(f.state.join("events.jsonl"), vec![b'X'; 131073]).unwrap();
    assert!(Server::open(&f.state).is_err());
}

#[test]
fn daemon_state_recovery_refuses_unrecognized_event_even_if_hash_valid() {
    let f = Fixture::new();
    fs::create_dir(&f.state).unwrap();
    let mut j = Journal::open(&f.state.join("events.jsonl")).unwrap();
    j.append(Value::object([
        ("type".into(), s("transaction.executing")),
        ("session".into(), s("s")),
        ("key".into(), s("k")),
        ("fingerprint".into(), s("h")),
        ("response".into(), Value::object([])),
    ]))
    .unwrap();
    drop(j);
    assert!(Server::open(&f.state).is_err());
}
