use std::{
    fs,
    path::PathBuf,
    process::Command,
    time::{Duration, SystemTime, UNIX_EPOCH},
};
use voder_runtime::{
    artifact_store::{sha256, ArtifactStore},
    environment::{EnvironmentPlan, Language},
    patch::{self, FilePatch, PatchOperation},
    paths,
    process::ProcessLimits,
    repository::Repository,
    rollback::{self},
    sandbox::{SandboxAvailability, SandboxBackend, SandboxPolicy},
    sandbox_bwrap::Bubblewrap,
    worktree::OwnedWorktree,
};

fn fresh(label: &str) -> PathBuf {
    let ns = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let p = std::env::temp_dir().join(format!(
        "voder-runtime-test-{label}-{}-{ns}",
        std::process::id()
    ));
    fs::create_dir(&p).unwrap();
    p
}
fn git(root: &std::path::Path, args: &[&str]) {
    let output = Command::new("git")
        .arg("-C")
        .arg(root)
        .args(args)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "git {:?}: {}",
        args,
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn reject_path_traversal_and_metadata_modification() {
    for bad in [
        "",
        "/etc/passwd",
        "../file",
        "src/../file",
        "src//main.rs",
        "./file",
        "src\\a",
        ".git/config",
        "src/.git/config",
        ".voder/keys",
        "a:\\b",
        "foo.",
        "foo\nbar",
    ] {
        assert!(paths::relative_path(bad).is_err(), "accepted {bad:?}");
    }
    assert!(paths::relative_path("src/lib.rs").is_ok());
}
#[test]
fn reject_symlinked_parent_and_file() {
    let root = fresh("symlink");
    fs::create_dir(root.join("src")).unwrap();
    #[cfg(unix)]
    {
        std::os::unix::fs::symlink("/etc", root.join("secret")).unwrap();
        assert!(paths::checked_destination(&root, "secret/passwd").is_err());
        std::os::unix::fs::symlink("/etc/passwd", root.join("src/linked")).unwrap();
        assert!(paths::checked_existing(&root, "src/linked").is_err());
    }
    fs::remove_dir_all(root).unwrap();
}
#[test]
fn git_worktree_mutation_does_not_touch_original() {
    let root = fresh("git");
    let repo = root.join("project");
    fs::create_dir(&repo).unwrap();
    git(&repo, &["init", "-q"]);
    fs::write(repo.join("main.rs"), "fn main(){}\n").unwrap();
    git(&repo, &["add", "main.rs"]);
    git(
        &repo,
        &[
            "-c",
            "user.email=dev@example.test",
            "-c",
            "user.name=Tester",
            "commit",
            "-qm",
            "init",
        ],
    );
    let r = Repository::inspect(&repo).unwrap();
    assert!(r.clean().unwrap());
    assert_eq!(
        r.inventory(&["missing.rs".into(), "main.rs".into()])
            .unwrap(),
        vec!["missing.rs"]
    );
    let original = r.fingerprint().unwrap();
    let w = OwnedWorktree::create(&r, &root.join("worktrees"), "task-1").unwrap();
    fs::write(w.path().join("main.rs"), "fn main(){panic!()}\n").unwrap();
    assert_ne!(
        voder_runtime::repository::fingerprint(w.path()).unwrap(),
        original
    );
    assert_eq!(r.fingerprint().unwrap(), original);
    w.finish().unwrap();
    assert!(!root.join("worktrees/task-1").exists());
    assert_eq!(r.fingerprint().unwrap(), original);
    fs::remove_dir_all(root).unwrap();
}
#[test]
fn dirty_repository_refuses_isolation_without_loss() {
    let root = fresh("dirty");
    let repo = root.join("src");
    fs::create_dir(&repo).unwrap();
    git(&repo, &["init", "-q"]);
    fs::write(repo.join("x"), "dirty").unwrap();
    // repository without HEAD cannot be taken over; don't wipe files.
    assert!(Repository::inspect(&repo).is_err());
    assert_eq!(fs::read_to_string(repo.join("x")).unwrap(), "dirty");
    fs::remove_dir_all(root).unwrap();
}
#[test]
fn artifact_integrity_and_pages() {
    let root = fresh("artifact");
    let store = ArtifactStore::open(&root).unwrap();
    let id = store.save(b"error: mismatch").unwrap();
    assert_eq!(store.page(&id, 7, 8).unwrap(), b"mismatch");
    assert!(store.page(&id, 0, 65537).is_err());
    fs::write(root.join(&id), "forged").unwrap();
    assert!(store.read(&id).is_err());
    assert!(store.read("../etc/passwd").is_err());
    fs::remove_dir_all(root).unwrap();
}
#[test]
fn patch_preimage_and_modify_rollback() {
    let root = fresh("patch");
    let artifacts = fresh("patchstore");
    let store = ArtifactStore::open(&artifacts).unwrap();
    fs::write(root.join("code.rs"), "bad").unwrap();
    let before = sha256(b"bad").unwrap();
    let mut s = rollback::snapshot(&root, "code.rs", &store).unwrap();
    assert!(patch::apply(
        &root,
        &FilePatch {
            relative: "code.rs".into(),
            operation: PatchOperation::Replace {
                old_sha256: sha256(b"wrong").unwrap(),
                new_bytes: b"fixed".to_vec()
            }
        }
    )
    .is_err());
    assert_eq!(fs::read(root.join("code.rs")).unwrap(), b"bad");
    patch::apply(
        &root,
        &FilePatch {
            relative: "code.rs".into(),
            operation: PatchOperation::Replace {
                old_sha256: before,
                new_bytes: b"fixed".to_vec(),
            },
        },
    )
    .unwrap();
    rollback::mark_after(&mut s, &root, &store).unwrap();
    rollback::restore(&root, &s, &store).unwrap();
    assert_eq!(fs::read(root.join("code.rs")).unwrap(), b"bad");
    fs::remove_dir_all(root).unwrap();
    fs::remove_dir_all(artifacts).unwrap();
}
#[test]
fn create_and_delete_can_be_recovered() {
    let root = fresh("created");
    let artifacts = fresh("created-store");
    let store = ArtifactStore::open(&artifacts).unwrap();
    let mut new_file = rollback::snapshot(&root, "new.rs", &store).unwrap();
    patch::apply(
        &root,
        &FilePatch {
            relative: "new.rs".into(),
            operation: PatchOperation::Create {
                new_bytes: b"new".to_vec(),
            },
        },
    )
    .unwrap();
    rollback::mark_after(&mut new_file, &root, &store).unwrap();
    rollback::restore(&root, &new_file, &store).unwrap();
    assert!(!root.join("new.rs").exists());
    fs::write(root.join("old.rs"), "old").unwrap();
    let mut deleted = rollback::snapshot(&root, "old.rs", &store).unwrap();
    patch::apply(
        &root,
        &FilePatch {
            relative: "old.rs".into(),
            operation: PatchOperation::Delete {
                old_sha256: sha256(b"old").unwrap(),
            },
        },
    )
    .unwrap();
    rollback::mark_after(&mut deleted, &root, &store).unwrap();
    rollback::restore(&root, &deleted, &store).unwrap();
    assert_eq!(fs::read(root.join("old.rs")).unwrap(), b"old");
    fs::remove_dir_all(root).unwrap();
    fs::remove_dir_all(artifacts).unwrap();
}
#[test]
fn rollback_rejects_divergent_workspace() {
    let root = fresh("divergent");
    let artifacts = fresh("divergent-store");
    let store = ArtifactStore::open(&artifacts).unwrap();
    fs::write(root.join("a"), b"first").unwrap();
    let mut s = rollback::snapshot(&root, "a", &store).unwrap();
    fs::write(root.join("a"), b"second").unwrap();
    rollback::mark_after(&mut s, &root, &store).unwrap();
    fs::write(root.join("a"), b"third").unwrap();
    assert!(rollback::restore(&root, &s, &store).is_err());
    assert_eq!(fs::read(root.join("a")).unwrap(), b"third");
    fs::remove_dir_all(root).unwrap();
    fs::remove_dir_all(artifacts).unwrap();
}
#[test]
fn environment_requires_real_project_files() {
    let root = fresh("env");
    assert!(EnvironmentPlan::inspect(&root, Language::Rust).is_err());
    fs::write(root.join("Cargo.toml"), "[package]\nname='x'\n").unwrap();
    assert!(EnvironmentPlan::inspect(&root, Language::Rust).is_ok());
    assert!(EnvironmentPlan::inspect(&root, Language::Node).is_err());
    fs::remove_dir_all(root).unwrap();
}
#[test]
fn sandbox_is_fail_closed_when_unavailable() {
    let root = fresh("sb");
    let policy = SandboxPolicy::new(
        root.clone(),
        ProcessLimits::new(Duration::from_secs(2), 1024, 512 * 1024 * 1024, 2).unwrap(),
    );
    let backend = Bubblewrap;
    if let SandboxAvailability::Unavailable(_) = backend.probe() {
        assert!(backend
            .execute(&policy, "/bin/echo", &["hello".into()], &root.join("logs"))
            .is_err());
        assert!(!root.join("logs/stdout.raw").exists());
    }
    let mut networked = policy.clone();
    networked.network = true;
    assert!(networked.validate().is_err());
    fs::remove_dir_all(root).unwrap();
}
#[test]
fn process_limits_reject_unbounded_values() {
    assert!(ProcessLimits::new(Duration::from_secs(0), 1024, 128 * 1024 * 1024, 2).is_err());
    assert!(ProcessLimits::new(Duration::from_secs(1), 0, 128 * 1024 * 1024, 2).is_err());
    assert!(ProcessLimits::new(Duration::from_secs(1), 1024, 0, 2).is_err());
}

#[test]
fn tool_registry_rejects_misrepresented_effects_and_duplicate_names() {
    use voder_core::types::ToolEffect;
    use voder_runtime::tool_registry::{ToolKind, ToolRegistry};
    let mut registry = ToolRegistry::new();
    assert!(registry
        .insert("patch", ToolKind::Patch, ToolEffect::Read)
        .is_err());
    registry
        .insert("patch", ToolKind::Patch, ToolEffect::Write)
        .unwrap();
    assert!(registry
        .insert("patch", ToolKind::Patch, ToolEffect::Write)
        .is_err());
    assert!(registry
        .insert("../../bad", ToolKind::ReadFile, ToolEffect::Read)
        .is_err());
    assert!(registry
        .insert("read", ToolKind::ReadFile, ToolEffect::Read)
        .is_ok());
}
#[test]
fn toolchain_mounts_are_admin_pinned_not_arbitrary_directories() {
    use voder_runtime::sandbox::ToolchainKind;
    let root = fresh("toolchain");
    let mut policy = SandboxPolicy::new(
        root.clone(),
        ProcessLimits::new(Duration::from_secs(2), 1024, 512 * 1024 * 1024, 2).unwrap(),
    );
    assert!(policy.add_toolchain(&root, ToolchainKind::Node).is_err());
    let tools = root.join(".voder-tools");
    fs::create_dir(&tools).unwrap();
    fs::create_dir(tools.join("node")).unwrap();
    assert!(policy.add_toolchain(&tools, ToolchainKind::Node).is_ok());
    assert!(policy.add_toolchain(&tools, ToolchainKind::Node).is_err());
    assert!(policy
        .add_toolchain(&tools, ToolchainKind::RustStable)
        .is_err());
    fs::remove_dir_all(root).unwrap();
}
#[test]
fn diagnostic_packets_are_evidence_not_verified_root_causes() {
    use voder_runtime::diagnostics::{prepare, DiagnosisStatus};
    let dir = fresh("diagnostics");
    let store = ArtifactStore::open(&dir).unwrap();
    let stdout = store.save(b"compile unsuccessful").unwrap();
    let stderr = store.save(b"error[E0425]: not found").unwrap();
    let p = prepare(
        &store,
        &stdout,
        &stderr,
        vec!["missing symbol import".into()],
        "inspect module exports".into(),
    )
    .unwrap();
    assert_eq!(p.status, DiagnosisStatus::NeedsInvestigation);
    assert!(p.excerpt.contains("E0425"));
    assert!(prepare(&store, &stdout, &stderr, vec![], "".into()).is_err());
    fs::remove_dir_all(dir).unwrap();
}
#[test]
fn evidence_objects_and_process_logs_are_private_to_owner() {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let dir = fresh("private-store");
        let store = ArtifactStore::open(&dir).unwrap();
        let id = store.save(b"potential secret").unwrap();
        assert_eq!(
            fs::metadata(&dir).unwrap().permissions().mode() & 0o777,
            0o700
        );
        assert_eq!(
            fs::metadata(dir.join(&id)).unwrap().permissions().mode() & 0o777,
            0o600
        );
        fs::remove_dir_all(dir).unwrap();
    }
}
