//! Non-destructive actual OS isolation probe. Run only after provisioning Bubblewrap.
//! Missing sandbox is a BLOCK; tests never use host execution as a fallback.
use std::{
    fs, process,
    time::{Duration, SystemTime, UNIX_EPOCH},
};
use voder_runtime::{
    sandbox::{SandboxAvailability, SandboxBackend, SandboxPolicy},
    sandbox_bwrap::Bubblewrap,
};
fn main() {
    let backend = Bubblewrap;
    match backend.probe() {
        SandboxAvailability::Unavailable(why) => {
            eprintln!("VODER_BWRAP_NEGATIVE_SECURITY=BLOCKED: {why}");
            process::exit(42);
        }
        SandboxAvailability::Available => println!("VODER_BWRAP_NAMESPACE_PROBE=PASS"),
    }
    let root = std::env::temp_dir().join(format!(
        "voder-sandbox-negative-{}-{}",
        process::id(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let work = root.join("work");
    let capture = root.join("logs");
    fs::create_dir_all(&work).unwrap();
    // Pretend git metadata; must be mounted read-only inside sandbox.
    fs::write(work.join(".git"), "gitdir: /not-mounted/metadata").unwrap();
    let secret = root.join("host-only-secret.txt");
    fs::write(&secret, "DO_NOT_EXPOSE_TO_SANDBOX").unwrap();
    let policy = SandboxPolicy::new(
        work.clone(),
        voder_runtime::process::ProcessLimits::new(
            Duration::from_secs(5),
            64 * 1024,
            512 * 1024 * 1024,
            5,
        )
        .unwrap(),
    );
    // Shell is confined to namespace; never run outside sandbox.
    let tests = [
        (
            "FILESYSTEM",
            "test ! -e /etc/passwd && test ! -e /root/.ssh && test ! -e /tmp/voder-nonexistent",
        ),
        (
            "WRITE_WORKTREE",
            "printf 'allowed' > /workspace/probe && test \"$(cat /workspace/probe)\" = allowed",
        ),
        (
            "GIT_METADATA",
            "if printf 'tamper' > /workspace/.git 2>/dev/null; then exit 1; else exit 0; fi",
        ),
    ];
    for (label, cmd) in tests {
        let slot = capture.join(label);
        match backend.execute(&policy, "/bin/sh", &["-c".into(), cmd.into()], &slot) {
            Ok(report) if report.exit_code() == Some(0) && !report.timed_out() => {
                println!("NEGATIVE_{label}=PASS")
            }
            Ok(report) => {
                eprintln!(
                    "NEGATIVE_{label}=FAIL rc={:?} stderr=...",
                    report.exit_code()
                );
                process::exit(1);
            }
            Err(e) => {
                eprintln!("NEGATIVE_{label}=FAIL {e}");
                process::exit(1);
            }
        }
    }
    // A second independent process probes network capability from the sandbox.
    if std::path::Path::new("/usr/bin/python3").exists() {
        let network_cmd="import socket\ns=socket.socket();s.settimeout(1)\ntry:\n s.connect(('1.1.1.1',80));raise SystemExit(5)\nexcept OSError:\n raise SystemExit(0)\n";
        let slot = capture.join("NETWORK");
        match backend.execute(
            &policy,
            "/usr/bin/python3",
            &["-c".into(), network_cmd.into()],
            &slot,
        ) {
            Ok(report) if report.exit_code() == Some(0) => println!("NEGATIVE_NETWORK=PASS"),
            other => {
                eprintln!("NEGATIVE_NETWORK=FAIL: {other:?}");
                process::exit(1);
            }
        }
    } else {
        eprintln!("NEGATIVE_NETWORK=BLOCKED: python3 not available in host image");
        process::exit(42);
    }
    if fs::read_to_string(&secret).unwrap() != "DO_NOT_EXPOSE_TO_SANDBOX" {
        eprintln!("HOST_DATA_CHANGED=FAIL");
        process::exit(1);
    }
    println!("VODER_BWRAP_NAMESPACE_SECURITY=PASS");
    // Keep seccomp and hostile-concurrent-race protection as separate mandatory product gates.
    println!("VODER_RUNTIME_PRODUCTION=BLOCKED: seccomp + race-hardening + crash atomicity not yet verified");
    // Do not present these namespace tests as full production-security certification.
    let _ = fs::remove_dir_all(root);
    process::exit(42);
}
