//! True Daemon request → trusted Node process → inert model receipt → journal test.
//! Runs with local test stub; no provider API key and no billable network call.
use std::{fs, path::Path, process::Command};
use voder_protocol::{
    envelope::{Envelope, FrameKind},
    json::Value,
};
use voderd::protocol_stdio::Server;
fn s(x: &str) -> Value {
    Value::String(x.into())
}
fn props(items: impl IntoIterator<Item = (String, Value)>) -> Value {
    Value::object(items)
}
fn request(
    id: &str,
    method: &str,
    session: Option<&str>,
    key: Option<&str>,
    body: Value,
) -> Envelope {
    let mut r = Envelope::request(id, method, body);
    r.session_id = session.map(str::to_owned);
    r.idempotency_key = key.map(str::to_owned);
    r
}
fn ok(srv: &mut Server, req: Envelope) -> Value {
    let out = srv.handle(req).expect("valid envelope");
    assert_eq!(out.frame_type, FrameKind::Response, "{out:?}");
    out.payload
}
fn setup_repo(home: &Path) -> std::path::PathBuf {
    let repo = home.join("repo");
    fs::create_dir(&repo).unwrap();
    assert!(Command::new("git")
        .args(["init", "-q"])
        .arg(&repo)
        .status()
        .unwrap()
        .success());
    fs::write(repo.join("README.md"), "sample\n").unwrap();
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
            "user.name=Voder",
            "-c",
            "user.email=voder@example.invalid",
            "commit",
            "-qm",
            "init"
        ])
        .status()
        .unwrap()
        .success());
    repo
}
fn init(srv: &mut Server) {
    ok(
        srv,
        request(
            "init",
            "initialize",
            None,
            None,
            props([
                ("minimum_version".into(), Value::Number("1".into())),
                ("maximum_version".into(), Value::Number("1".into())),
                ("client_name".into(), s("seventh-test")),
                ("requested_features".into(), Value::Array(vec![])),
            ]),
        ),
    );
}
fn create(srv: &mut Server, repo: &Path) -> String {
    let output = ok(
        srv,
        request(
            "create",
            "session.create",
            None,
            Some("sess-1"),
            props([("project_root".into(), s(repo.to_str().unwrap()))]),
        ),
    );
    output.get("session_id").unwrap().as_str().unwrap().into()
}
fn submit(srv: &mut Server, sid: &str, reqid: &str, refs: Vec<Value>) -> Value {
    ok(
        srv,
        request(
            reqid,
            "task.submit",
            Some(sid),
            Some("task-key"),
            props([
                ("instruction".into(), s("Fix failing test")),
                ("context_refs".into(), Value::Array(refs)),
            ]),
        ),
    )
}
#[test]
fn provider_submission_is_durable_non_authoritative_idempotent_and_recoverable() {
    let node_output = Command::new("which")
        .arg("node")
        .output()
        .expect("Node 22 required for integration test");
    assert!(
        node_output.status.success(),
        "Node is required for integration test"
    );
    // Respect the production bridge's no-symlink policy: test PATH may
    // contain a version-manager or AutoDL symlink to a pinned Node binary.
    let node = fs::canonicalize(String::from_utf8(node_output.stdout).unwrap().trim())
        .expect("resolve Node to a trusted regular executable file");
    assert!(node.is_absolute() && node.is_file()
        && !fs::symlink_metadata(&node).unwrap().file_type().is_symlink());
    let home = std::env::temp_dir().join(format!("voder7-contract-{}", std::process::id()));
    fs::create_dir_all(&home).unwrap();
    let repo = setup_repo(&home);
    let script = home.join("bridge.mjs");
    fs::write(&script,r#"import fs from 'node:fs';
const [input,output] = process.argv.slice(2);
const task = JSON.parse(fs.readFileSync(input,'utf8'));
fs.writeFileSync(output,JSON.stringify({protocol:'voder-provider-bridge-v1',runId:task.runId,providerId:'mock',model:'mock-model',proposalText:'inert fix suggestion',proposedToolCalls:[],finishReason:'stop',deliveryAllowed:false,executionAuthorized:false}),{mode:0o600,flag:'wx'});
"#).unwrap();
    let state = home.join("state");
    // Dedicated test binary, no other tests mutate this process environment.
    std::env::set_var("VODER_PROVIDER_NODE", &node);
    std::env::set_var("VODER_PROVIDER_CLI", &script);
    let mut srv = Server::open(&state).unwrap();
    init(&mut srv);
    let sid = create(&mut srv, &repo);
    let ack = submit(&mut srv, &sid, "task-1", vec![]);
    assert_eq!(
        ack.get("status").and_then(Value::as_str),
        Some("MODEL_PENDING")
    );
    assert_eq!(ack.get("delivery_allowed"), Some(&Value::Bool(false)));
    let run = ack.get("run_id").unwrap().as_str().unwrap();
    let status = ok(
        &mut srv,
        request(
            "get",
            "state.get",
            Some(&sid),
            None,
            props([("run_id".into(), s(run))]),
        ),
    );
    let Value::Array(rows) = status.get("runs").unwrap() else {
        panic!("missing runs");
    };
    assert_eq!(
        rows[0].get("status").and_then(Value::as_str),
        Some("PROPOSED")
    );
    let digest = rows[0].get("proposal_digest").unwrap().as_str().unwrap();
    assert_eq!(digest.len(), 64);
    let page = ok(
        &mut srv,
        request(
            "artifact",
            "artifact.read",
            Some(&sid),
            None,
            props([
                ("digest".into(), s(&format!("sha256:{digest}"))),
                ("offset".into(), Value::Number("0".into())),
                ("limit".into(), Value::Number("1024".into())),
            ]),
        ),
    );
    assert_eq!(page.get("encoding").and_then(Value::as_str), Some("hex"));
    assert!(page.get("data_hex").unwrap().as_str().unwrap().len() > 20);
    assert_eq!(srv.event_count(), 3);
    drop(srv);
    let mut resumed = Server::open(&state).unwrap();
    init(&mut resumed);
    let again = submit(&mut resumed, &sid, "task-2", vec![]);
    assert_eq!(again, ack);
    assert_eq!(
        resumed.event_count(),
        3,
        "idempotent replay must not repeat provider request"
    );
    drop(resumed);
    std::env::remove_var("VODER_PROVIDER_NODE");
    std::env::remove_var("VODER_PROVIDER_CLI");
    // Test temp resources are owned by the test; never touch any real workspace.
    fs::remove_dir_all(&home).unwrap();
}
