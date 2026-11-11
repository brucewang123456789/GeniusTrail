//! Crash-recoverable, single-file patch intent ledger for an already-authorized
//! VGR write action. This module does not grant execution or delivery rights.
//!
//! The daemon must retain the exclusive journal/worktree writer lock and must
//! inspect journal state before invoking `recover_uncommitted`. Recovery only
//! returns a file to its known preimage; an unknown modification fails closed.
//! The WAL is deliberately outside the workspace and is never removed here.
use crate::{
    artifact_store::{sha256, ArtifactStore},
    patch::{self, FilePatch, PatchOperation, PatchResult},
    paths, Result, RuntimeError,
};
use std::{
    fs::{self, File, OpenOptions},
    io::Write,
    path::{Path, PathBuf},
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RecoveryOutcome {
    /// The mutation was never applied, or was already reverted.
    AlreadyAtPreimage,
    /// Its exact expected postimage was observed, then safely reversed.
    RolledBack,
}

#[derive(Debug, Clone)]
struct Entry {
    action_key: String,
    relative: String,
    before: Option<String>,
    after: Option<String>,
}

fn hex_or_dash(x: &Option<String>) -> &str {
    x.as_deref().unwrap_or("-")
}

fn parse_hash(raw: &str) -> Result<Option<String>> {
    if raw == "-" {
        return Ok(None);
    }
    if raw.len() != 64 || !raw.bytes().all(|c| c.is_ascii_hexdigit()) {
        return Err(RuntimeError::Integrity("invalid durable patch hash".into()));
    }
    Ok(Some(raw.to_ascii_lowercase()))
}

fn serialize(entry: &Entry) -> Result<Vec<u8>> {
    let fields = format!(
        "VODER_DURABLE_PATCH_V1\n{}\n{}\n{}\n{}\n",
        entry.action_key,
        entry.relative,
        hex_or_dash(&entry.before),
        hex_or_dash(&entry.after),
    );
    let signature = sha256(fields.as_bytes())?;
    Ok(format!("{fields}{signature}\n").into_bytes())
}

fn parse(bytes: &[u8]) -> Result<Entry> {
    if bytes.len() > 8192 {
        return Err(RuntimeError::Integrity(
            "oversized durable patch record".into(),
        ));
    }
    let data = std::str::from_utf8(bytes)
        .map_err(|_| RuntimeError::Integrity("non-UTF8 durable patch record".into()))?;
    let lines: Vec<&str> = data.split('\n').collect();
    if lines.len() != 7 || !lines[6].is_empty() || lines[0] != "VODER_DURABLE_PATCH_V1" {
        return Err(RuntimeError::Integrity(
            "malformed durable patch record".into(),
        ));
    }
    let raw_content = format!(
        "{}\n{}\n{}\n{}\n{}\n",
        lines[0], lines[1], lines[2], lines[3], lines[4]
    );
    if sha256(raw_content.as_bytes())? != lines[5] {
        return Err(RuntimeError::Integrity(
            "durable patch record checksum mismatch".into(),
        ));
    }
    validate_action_key(lines[1])?;
    paths::relative_path(lines[2])?;
    Ok(Entry {
        action_key: lines[1].to_owned(),
        relative: lines[2].to_owned(),
        before: parse_hash(lines[3])?,
        after: parse_hash(lines[4])?,
    })
}

fn validate_action_key(key: &str) -> Result<()> {
    if key.is_empty()
        || key.len() > 256
        || !key
            .bytes()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, b'-' | b'_' | b'.' | b':'))
    {
        return Err(RuntimeError::Invalid(
            "invalid VGR durable action key".into(),
        ));
    }
    Ok(())
}

fn open_wal_directory(root: &Path, wal_root: &Path) -> Result<PathBuf> {
    let workspace = root.canonicalize()?;
    if !workspace.is_dir() {
        return Err(RuntimeError::Invalid(
            "write workspace is not a directory".into(),
        ));
    }
    // Fail before creating anything under an attacker-controlled checkout.
    let wal_parent = wal_root
        .parent()
        .ok_or(RuntimeError::Unauthorized("WAL parent missing"))?;
    let parent = wal_parent.canonicalize()?;
    if parent.starts_with(&workspace)
        || workspace.starts_with(
            parent.join(
                wal_root
                    .file_name()
                    .ok_or(RuntimeError::Unauthorized("WAL name missing"))?,
            ),
        )
    {
        return Err(RuntimeError::Unauthorized("WAL may not overlap worktree"));
    }
    if wal_root.exists() || fs::symlink_metadata(wal_root).is_ok() {
        let meta = fs::symlink_metadata(wal_root)?;
        if !meta.is_dir() || meta.file_type().is_symlink() {
            return Err(RuntimeError::Unauthorized("WAL is not a real directory"));
        }
    } else {
        fs::create_dir(wal_root)?;
        File::open(&parent)?.sync_all()?;
    }
    let wal = wal_root.canonicalize()?;
    if wal.starts_with(&workspace) || workspace.starts_with(&wal) {
        return Err(RuntimeError::Unauthorized("WAL/worktree directory overlap"));
    }
    Ok(wal)
}

fn entry_path(dir: &Path, action_key: &str) -> Result<PathBuf> {
    validate_action_key(action_key)?;
    Ok(dir.join(format!(
        "{}.wal",
        sha256(format!("voder-wal-key-v1\0{action_key}").as_bytes())?
    )))
}

fn measured(root: &Path, relative: &str) -> Result<Option<String>> {
    let path = paths::checked_destination(root, relative)?;
    match fs::symlink_metadata(&path) {
        Ok(meta) if meta.is_file() && !meta.file_type().is_symlink() => {
            Ok(Some(crate::artifact_store::sha256_file(&path)?))
        }
        Ok(_) => Err(RuntimeError::UnsafePath("special patch target".into())),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(e.into()),
    }
}

fn fsync_parent(root: &Path, relative: &str) -> Result<()> {
    let file = paths::checked_destination(root, relative)?;
    File::open(
        file.parent()
            .ok_or(RuntimeError::Unauthorized("missing patch parent"))?,
    )?
    .sync_all()?;
    Ok(())
}

/// Apply exactly one previously authorized mutation. A WAL record containing
/// the original content-addressed preimage is durable before any mutation.
/// A repeated action key is *always* refused, even after a crash/recovery.
/// The caller holds the daemon writer lock and a consumed durable dispatch permit.
pub fn apply(
    root: &Path,
    wal_root: &Path,
    action_key: &str,
    change: &FilePatch,
    store: &ArtifactStore,
) -> Result<PatchResult> {
    paths::relative_path(&change.relative)?;
    let dir = open_wal_directory(root, wal_root)?;
    let path = entry_path(&dir, action_key)?;
    let before = measured(root, &change.relative)?;
    let after = match &change.operation {
        PatchOperation::Create { new_bytes } if before.is_none() => {
            Some(store.save(new_bytes)?)
        }
        PatchOperation::Replace {
            old_sha256,
            new_bytes,
        } if before.as_deref() == Some(old_sha256) => {
            Some(store.save(new_bytes)?)
        }
        PatchOperation::Delete { old_sha256 } if before.as_deref() == Some(old_sha256) => None,
        _ => {
            return Err(RuntimeError::Integrity(
                "durable patch preimage mismatch".into(),
            ))
        }
    };
    if let Some(pre) = &before {
        let bytes = fs::read(paths::checked_existing(root, &change.relative)?)?;
        if store.save(&bytes)? != *pre {
            return Err(RuntimeError::Integrity(
                "durable patch preimage changed".into(),
            ));
        }
    }
    let record = serialize(&Entry {
        action_key: action_key.to_owned(),
        relative: change.relative.clone(),
        before,
        after: after.clone(),
    })?;
    let mut opts = OpenOptions::new();
    opts.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        opts.mode(0o600);
        opts.custom_flags(0o400000); // O_NOFOLLOW on Linux
    }
    let mut file = opts.open(path)?;
    file.write_all(&record)?;
    file.sync_all()?;
    File::open(&dir)?.sync_all()?;
    // An interrupted operation is intentionally left for restart reconciliation.
    let result = patch::apply(root, change)?;
    fsync_parent(root, &change.relative)?;
    if measured(root, &change.relative)? != after {
        return Err(RuntimeError::Integrity(
            "durable patch postimage mismatch".into(),
        ));
    }
    Ok(result)
}

/// Verify that the only difference from the prepared immutable full-workspace
/// snapshot is the particular file described by this durable WAL. This is
/// a preflight, not the actual rollback; it must happen under the daemon's
/// exclusive writer lock before any workspace bytes are modified.
pub fn verify_recovery_scope(
    root: &Path,
    wal_root: &Path,
    action_key: &str,
    prepared_workspace_digest: &str,
) -> Result<()> {
    if prepared_workspace_digest.len() != 64
        || !prepared_workspace_digest
            .bytes()
            .all(|c| c.is_ascii_hexdigit())
    {
        return Err(RuntimeError::Integrity(
            "invalid prepared workspace digest".into(),
        ));
    }
    let dir = open_wal_directory(root, wal_root)?;
    let path = entry_path(&dir, action_key)?;
    let meta = fs::symlink_metadata(&path)?;
    if !meta.is_file() || meta.file_type().is_symlink() {
        return Err(RuntimeError::Integrity("unsafe WAL record".into()));
    }
    let entries = parse_entries(&fs::read(&path)?)?;
    let mut files = crate::repository::snapshot_files(root)?;
    for entry in entries {
    if entry.action_key != action_key {
        return Err(RuntimeError::Integrity("WAL action mismatch".into()));
    }
    let current = measured(root, &entry.relative)?;
    if current != entry.before && current != entry.after {
        return Err(RuntimeError::Integrity(
            "write target diverged from WAL".into(),
        ));
    }
    match &entry.before {
        Some(preimage) => {
            files.insert(entry.relative.clone(), format!("F:{preimage}"));
        }
        None => {
            files.remove(&entry.relative);
        }
    }
    }
    let candidate = crate::repository::hash_snapshot(&files)?;
    if candidate != prepared_workspace_digest {
        return Err(RuntimeError::Integrity(
            "unrelated workspace changes block durable recovery".into(),
        ));
    }
    Ok(())
}

/// Read the durable record and revert ONLY if the workspace equals the exact
/// expected postimage. If the write was not applied, return a no-op. A divergent
/// file is never overwritten. Call ONLY after native Journal reconstruction
/// independently proves the action has NOT been finally committed.
pub fn recover_uncommitted(
    root: &Path,
    wal_root: &Path,
    action_key: &str,
    store: &ArtifactStore,
) -> Result<RecoveryOutcome> {
    let dir = open_wal_directory(root, wal_root)?;
    let path = entry_path(&dir, action_key)?;
    let meta = fs::symlink_metadata(&path)?;
    if !meta.is_file() || meta.file_type().is_symlink() {
        return Err(RuntimeError::Integrity("unsafe WAL record".into()));
    }
    let entries = parse_entries(&fs::read(&path)?)?;
    // Validate EVERY member before the first rollback; unknown user bytes block the whole operation.
    for entry in &entries {
        if entry.action_key != action_key { return Err(RuntimeError::Integrity("batch action mismatch".into())); }
        let current=measured(root,&entry.relative)?;
        if current != entry.before && current != entry.after { return Err(RuntimeError::Integrity("divergent batch member".into())); }
    }
    let mut changed=false;
    for entry in entries.iter().rev() {
        let current=measured(root,&entry.relative)?;
        if current==entry.before { continue; }
        if current!=entry.after { return Err(RuntimeError::Integrity("batch changed during recovery".into())); }
        let patch=match (&entry.before,&entry.after) {
            (Some(before),Some(after))=>FilePatch{relative:entry.relative.clone(),operation:PatchOperation::Replace{old_sha256:after.clone(),new_bytes:store.read(before)?}},
            (Some(before),None)=>FilePatch{relative:entry.relative.clone(),operation:PatchOperation::Create{new_bytes:store.read(before)?}},
            (None,Some(after))=>FilePatch{relative:entry.relative.clone(),operation:PatchOperation::Delete{old_sha256:after.clone()}},
            _=>continue,
        };
        patch::apply(root,&patch)?; fsync_parent(root,&entry.relative)?;
        if measured(root,&entry.relative)?!=entry.before {return Err(RuntimeError::Integrity("batch rollback postcheck".into()));}
        changed=true;
    }
    Ok(if changed {RecoveryOutcome::RolledBack} else {RecoveryOutcome::AlreadyAtPreimage})
}

fn parse_entries(bytes:&[u8])->Result<Vec<Entry>> {
    if !bytes.starts_with(b"VODER_DURABLE_BATCH_V1\n") { return Ok(vec![parse(bytes)?]); }
    if bytes.len()>65536 {return Err(RuntimeError::Integrity("batch WAL size".into()));}
    let text=std::str::from_utf8(bytes).map_err(|_|RuntimeError::Integrity("batch WAL UTF8".into()))?;
    let lines:Vec<&str>=text.split('\n').collect();
    let count=lines.get(1).and_then(|x|x.parse::<usize>().ok()).ok_or(RuntimeError::Integrity("batch count".into()))?;
    if !(2..=16).contains(&count)||lines.len()!=count*6+4||lines.last()!=Some(&"") {return Err(RuntimeError::Integrity("batch WAL shape".into()));}
    let body=format!("{}\n",lines[..lines.len()-2].join("\n"));
    if sha256(body.as_bytes())?!=lines[lines.len()-2] {return Err(RuntimeError::Integrity("batch WAL checksum".into()));}
    let mut out=Vec::new();let mut seen=std::collections::BTreeSet::new();
    for chunk in lines[2..2+count*6].chunks(6) {
        let entry=parse(format!("{}\n",chunk.join("\n")).as_bytes())?;
        if !seen.insert(entry.relative.clone()) {return Err(RuntimeError::Integrity("duplicate batch target".into()));}
        out.push(entry);
    }
    if out.iter().any(|e|e.action_key!=out[0].action_key) {return Err(RuntimeError::Integrity("batch identity mismatch".into()));}
    Ok(out)
}

/// One dispatch permit and ONE fsynced WAL cover the complete 2..16 member action.
/// No intermediate member can be observed/admitted/committed as a separate VGR action.
pub fn apply_batch(root:&Path,wal_root:&Path,action_key:&str,changes:&[FilePatch],store:&ArtifactStore)->Result<Vec<PatchResult>> {
    if !(2..=16).contains(&changes.len()) {return Err(RuntimeError::Invalid("batch needs 2..16 files".into()));}
    let dir=open_wal_directory(root,wal_root)?;let path=entry_path(&dir,action_key)?;
    let mut seen=std::collections::BTreeSet::new();let mut entries=Vec::new();
    for change in changes {
        paths::relative_path(&change.relative)?;
        if !seen.insert(change.relative.clone()) {return Err(RuntimeError::Invalid("duplicate batch path".into()));}
        let before=measured(root,&change.relative)?;
        let after=match &change.operation {
            PatchOperation::Replace{old_sha256,new_bytes} if before.as_deref()==Some(old_sha256)&&!new_bytes.is_empty()=>Some(store.save(new_bytes)?),
            _=>return Err(RuntimeError::Integrity("batch requires bounded existing-file replacements and exact preimages".into())),
        };
        let bytes=fs::read(paths::checked_existing(root,&change.relative)?)?;
        if store.save(&bytes)?!=before.clone().unwrap_or_default() {return Err(RuntimeError::Integrity("batch preimage race".into()));}
        entries.push(Entry{action_key:action_key.to_owned(),relative:change.relative.clone(),before,after});
    }
    let mut body=format!("VODER_DURABLE_BATCH_V1\n{}\n",entries.len()).into_bytes();
    for e in &entries {body.extend(serialize(e)?);}
    let checksum=sha256(&body)?;body.extend(format!("{checksum}\n").as_bytes());
    let mut opts=OpenOptions::new();opts.write(true).create_new(true);
    #[cfg(unix)] {use std::os::unix::fs::OpenOptionsExt;opts.mode(0o600).custom_flags(0o400000);}
    let mut f=opts.open(path)?;f.write_all(&body)?;f.sync_all()?;File::open(&dir)?.sync_all()?;
    let apply_result=(||->Result<Vec<PatchResult>> {
        let mut results=Vec::new();
        for (index,change) in changes.iter().enumerate() {
            results.push(patch::apply(root,change)?);fsync_parent(root,&change.relative)?;
            #[cfg(test)] if index==0 && std::env::var_os("VODER_BATCH_TEST_CRASH").as_deref()==Some(root.as_os_str()) {
                use std::io::Write;println!("BATCH_FIRST_MEMBER_DURABLE");std::io::stdout().flush()?;loop{std::thread::sleep(std::time::Duration::from_millis(100));}
            }
            if measured(root,&change.relative)?!=entries[index].after {return Err(RuntimeError::Integrity("batch postimage".into()));}
        }
        Ok(results)
    })();
    match apply_result {
        Ok(results)=>Ok(results),
        Err(error)=>{recover_uncommitted(root,wal_root,action_key,store)?;Err(error)}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn fresh(tag: &str) -> PathBuf {
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "voder-durable-{tag}-{}-{stamp}",
            std::process::id()
        ));
        fs::create_dir(&path).unwrap();
        path
    }
    struct Fixture {
        root: PathBuf,
        wal_parent: PathBuf,
        store_parent: PathBuf,
        store: ArtifactStore,
    }
    impl Fixture {
        fn new(label: &str) -> Self {
            let root = fresh(&format!("root-{label}"));
            let wal_parent = fresh(&format!("wal-{label}"));
            let store_parent = fresh(&format!("store-{label}"));
            let store = ArtifactStore::open(&store_parent).unwrap();
            Self {
                root,
                wal_parent,
                store_parent,
                store,
            }
        }
        fn wal(&self) -> PathBuf {
            self.wal_parent.join("tx")
        }
    }
    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.root);
            let _ = fs::remove_dir_all(&self.wal_parent);
            let _ = fs::remove_dir_all(&self.store_parent);
        }
    }

    #[test]
    fn file_flex_large_replace_recovers_exact_preimage_and_refuses_replay() {
        let f=Fixture::new("large-flex");let before=vec![b'a';9*1024*1024];let after=vec![b'b';9*1024*1024+1];
        fs::write(f.root.join("large.rs"),&before).unwrap();
        let change=FilePatch{relative:"large.rs".into(),operation:PatchOperation::Replace{old_sha256:sha256(&before).unwrap(),new_bytes:after.clone()}};
        apply(&f.root,&f.wal(),"s:r:large",&change,&f.store).unwrap();
        assert_eq!(fs::read(f.root.join("large.rs")).unwrap(),after);
        assert!(apply(&f.root,&f.wal(),"s:r:large",&change,&f.store).is_err());
        assert_eq!(recover_uncommitted(&f.root,&f.wal(),"s:r:large",&f.store).unwrap(),RecoveryOutcome::RolledBack);
        assert_eq!(fs::read(f.root.join("large.rs")).unwrap(),before);
    }

    #[test]
    fn durable_replace_replays_identically_and_rejects_action_reuse() {
        let f = Fixture::new("replace");
        fs::write(f.root.join("test.rs"), b"incorrect").unwrap();
        let change = FilePatch {
            relative: "test.rs".into(),
            operation: PatchOperation::Replace {
                old_sha256: sha256(b"incorrect").unwrap(),
                new_bytes: b"correct".to_vec(),
            },
        };
        apply(&f.root, &f.wal(), "session:run:action", &change, &f.store).unwrap();
        assert_eq!(fs::read(f.root.join("test.rs")).unwrap(), b"correct");
        assert!(apply(&f.root, &f.wal(), "session:run:action", &change, &f.store).is_err());
        assert_eq!(
            recover_uncommitted(&f.root, &f.wal(), "session:run:action", &f.store).unwrap(),
            RecoveryOutcome::RolledBack
        );
        assert_eq!(fs::read(f.root.join("test.rs")).unwrap(), b"incorrect");
        assert_eq!(
            recover_uncommitted(&f.root, &f.wal(), "session:run:action", &f.store).unwrap(),
            RecoveryOutcome::AlreadyAtPreimage
        );
        assert!(apply(&f.root, &f.wal(), "session:run:action", &change, &f.store).is_err());
    }

    #[test]
    fn durable_create_and_delete_are_reversible() {
        let f = Fixture::new("create-delete");
        let create = FilePatch {
            relative: "new.rs".into(),
            operation: PatchOperation::Create {
                new_bytes: b"created".to_vec(),
            },
        };
        apply(&f.root, &f.wal(), "s:r:create", &create, &f.store).unwrap();
        assert_eq!(
            recover_uncommitted(&f.root, &f.wal(), "s:r:create", &f.store).unwrap(),
            RecoveryOutcome::RolledBack
        );
        assert!(!f.root.join("new.rs").exists());
        fs::write(f.root.join("old.rs"), b"original").unwrap();
        let delete = FilePatch {
            relative: "old.rs".into(),
            operation: PatchOperation::Delete {
                old_sha256: sha256(b"original").unwrap(),
            },
        };
        apply(&f.root, &f.wal(), "s:r:delete", &delete, &f.store).unwrap();
        assert!(!f.root.join("old.rs").exists());
        assert_eq!(
            recover_uncommitted(&f.root, &f.wal(), "s:r:delete", &f.store).unwrap(),
            RecoveryOutcome::RolledBack
        );
        assert_eq!(fs::read(f.root.join("old.rs")).unwrap(), b"original");
    }

    #[test]
    fn recovery_rejects_external_modification_without_overwriting() {
        let f = Fixture::new("divergent");
        fs::write(f.root.join("code.rs"), b"before").unwrap();
        let change = FilePatch {
            relative: "code.rs".into(),
            operation: PatchOperation::Replace {
                old_sha256: sha256(b"before").unwrap(),
                new_bytes: b"after".to_vec(),
            },
        };
        apply(&f.root, &f.wal(), "s:r:divergent", &change, &f.store).unwrap();
        fs::write(f.root.join("code.rs"), b"unrelated change").unwrap();
        assert!(recover_uncommitted(&f.root, &f.wal(), "s:r:divergent", &f.store).is_err());
        assert_eq!(
            fs::read(f.root.join("code.rs")).unwrap(),
            b"unrelated change"
        );
    }

    #[test]
    fn broken_wal_checksum_is_never_accepted() {
        let f = Fixture::new("tamper");
        let change = FilePatch {
            relative: "new.rs".into(),
            operation: PatchOperation::Create {
                new_bytes: b"after".to_vec(),
            },
        };
        apply(&f.root, &f.wal(), "s:r:tamper", &change, &f.store).unwrap();
        let path = entry_path(&f.wal(), "s:r:tamper").unwrap();
        let mut record = fs::read(&path).unwrap();
        record[9] ^= 1;
        fs::write(&path, &record).unwrap();
        assert!(recover_uncommitted(&f.root, &f.wal(), "s:r:tamper", &f.store).is_err());
        assert_eq!(fs::read(f.root.join("new.rs")).unwrap(), b"after");
    }

    #[test]
    fn write_ahead_log_inside_worktree_is_rejected() {
        let f = Fixture::new("escape");
        let change = FilePatch {
            relative: "new.rs".into(),
            operation: PatchOperation::Create {
                new_bytes: b"write".to_vec(),
            },
        };
        assert!(apply(
            &f.root,
            &f.root.join("state"),
            "s:r:escape",
            &change,
            &f.store
        )
        .is_err());
        assert!(!f.root.join("new.rs").exists());
        assert!(!f.root.join("state").exists());
    }
}

#[cfg(test)]
mod batch_tests {
 use super::*;
 fn setup()->(PathBuf,PathBuf,ArtifactStore){
  let base=std::env::temp_dir().join(format!("voder-batch-{}-{}",std::process::id(),std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()));
  fs::create_dir(&base).unwrap();let root=base.join("repo");fs::create_dir(&root).unwrap();fs::create_dir(root.join("src")).unwrap();fs::create_dir(root.join("other")).unwrap();
  fs::write(root.join("src/a"),b"old-a").unwrap();fs::write(root.join("other/b"),b"old-b").unwrap();fs::write(root.join("canary"),b"keep").unwrap();let store=ArtifactStore::open(&base.join("artifacts")).unwrap();(root,base.join("wal"),store)
 }
 fn patches()->Vec<FilePatch>{vec![FilePatch{relative:"other/b".into(),operation:PatchOperation::Replace{old_sha256:sha256(b"old-b").unwrap(),new_bytes:b"new-b".to_vec()}},FilePatch{relative:"src/a".into(),operation:PatchOperation::Replace{old_sha256:sha256(b"old-a").unwrap(),new_bytes:b"new-a".to_vec()}}]}
 #[test]fn batch_roundtrip_across_directories_and_no_replay(){let (r,w,s)=setup();let before=crate::repository::hash_snapshot(&crate::repository::snapshot_files(&r).unwrap()).unwrap();let changes=patches();apply_batch(&r,&w,"s:r:a",&changes,&s).unwrap();assert_eq!(fs::read(r.join("src/a")).unwrap(),b"new-a");verify_recovery_scope(&r,&w,"s:r:a",&before).unwrap();assert_eq!(recover_uncommitted(&r,&w,"s:r:a",&s).unwrap(),RecoveryOutcome::RolledBack);assert_eq!(fs::read(r.join("other/b")).unwrap(),b"old-b");assert!(apply_batch(&r,&w,"s:r:a",&changes,&s).is_err());}
 #[test]fn stale_later_member_never_changes_first(){let(r,w,s)=setup();let mut c=patches();if let PatchOperation::Replace{old_sha256,..}=&mut c[1].operation{*old_sha256="0".repeat(64)}assert!(apply_batch(&r,&w,"s:r:b",&c,&s).is_err());assert_eq!(fs::read(r.join("other/b")).unwrap(),b"old-b");}
 #[test]fn partial_batch_crash_is_recovered_from_complete_manifest(){let(r,w,s)=setup();apply_batch(&r,&w,"s:r:c",&patches(),&s).unwrap();fs::write(r.join("src/a"),b"old-a").unwrap();assert_eq!(recover_uncommitted(&r,&w,"s:r:c",&s).unwrap(),RecoveryOutcome::RolledBack);assert_eq!(fs::read(r.join("other/b")).unwrap(),b"old-b");}
 #[test]fn foreign_change_blocks_all_rollback(){let(r,w,s)=setup();apply_batch(&r,&w,"s:r:d",&patches(),&s).unwrap();fs::write(r.join("src/a"),b"user-edit").unwrap();assert!(recover_uncommitted(&r,&w,"s:r:d",&s).is_err());assert_eq!(fs::read(r.join("other/b")).unwrap(),b"new-b");assert_eq!(fs::read(r.join("src/a")).unwrap(),b"user-edit");}
 #[test]fn unrelated_workspace_edit_blocks_scope_recovery(){let(r,w,s)=setup();let before=crate::repository::hash_snapshot(&crate::repository::snapshot_files(&r).unwrap()).unwrap();apply_batch(&r,&w,"s:r:e",&patches(),&s).unwrap();fs::write(r.join("canary"),b"foreign").unwrap();assert!(verify_recovery_scope(&r,&w,"s:r:e",&before).is_err());}
 #[test]fn batch_checksum_tamper_and_duplicate_rejected(){let(r,w,s)=setup();let c=patches();assert!(apply_batch(&r,&w,"s:r:f",&[c[0].clone(),c[0].clone()],&s).is_err());apply_batch(&r,&w,"s:r:g",&c,&s).unwrap();let file=entry_path(&w,"s:r:g").unwrap();let mut data=fs::read(&file).unwrap();data[25]^=1;fs::write(&file,data).unwrap();assert!(recover_uncommitted(&r,&w,"s:r:g",&s).is_err());}
 #[test]fn batch_crash_child(){if let Some(root)=std::env::var_os("VODER_BATCH_TEST_CRASH") {let r=PathBuf::from(root);let base=r.parent().unwrap();let store=ArtifactStore::open(&base.join("artifacts")).unwrap();apply_batch(&r,&base.join("wal"),"s:r:crash",&patches(),&store).unwrap();}}
 #[test]fn killed_native_batch_restores_full_prepared_snapshot(){
  use std::io::{BufRead,BufReader};use std::process::{Command,Stdio};
  let(r,w,s)=setup();let before=crate::repository::hash_snapshot(&crate::repository::snapshot_files(&r).unwrap()).unwrap();
  let mut child=Command::new(std::env::current_exe().unwrap()).args(["--exact","durable_write::batch_tests::batch_crash_child","--nocapture"]).env("VODER_BATCH_TEST_CRASH",&r).stdout(Stdio::piped()).stderr(Stdio::null()).spawn().unwrap();
  let stdout=child.stdout.take().unwrap();let(tx,rx)=std::sync::mpsc::channel();let reader=std::thread::spawn(move||{for line in BufReader::new(stdout).lines(){if line.unwrap_or_default().contains("BATCH_FIRST_MEMBER_DURABLE"){let _=tx.send(());break;}}});
  let ready=rx.recv_timeout(std::time::Duration::from_secs(5)).is_ok();let _=child.kill();let status=child.wait().unwrap();reader.join().unwrap();assert!(ready);assert!(!status.success());
  assert_eq!(fs::read(r.join("other/b")).unwrap(),b"new-b");assert_eq!(fs::read(r.join("src/a")).unwrap(),b"old-a");verify_recovery_scope(&r,&w,"s:r:crash",&before).unwrap();recover_uncommitted(&r,&w,"s:r:crash",&s).unwrap();assert_eq!(crate::repository::hash_snapshot(&crate::repository::snapshot_files(&r).unwrap()).unwrap(),before);
 }

}
