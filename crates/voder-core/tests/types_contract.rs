use voder_core::types::*;

#[test]
fn opaque_ids_are_distinct_and_stable() {
    let task = TaskId::parse("task:001").unwrap();
    let action = ActionId::parse("task:001").unwrap();
    assert_eq!(task.as_str(), action.as_str());
    assert_eq!(task.to_string(), "task:001");
    // Not interchangeable at compile time: see the compile_fail documentation.
}

#[test]
fn ids_reject_empty_control_whitespace_and_overlong_values() {
    for value in ["", "  ", "\tA", "A\nB", " A", "A ", "\0"] {
        assert!(TaskId::parse(value).is_err(), "accepted invalid {value:?}");
    }
    assert!(TaskId::parse("x".repeat(MAX_ID_BYTES)).is_ok());
    assert!(TaskId::parse("x".repeat(MAX_ID_BYTES + 1)).is_err());
}

#[test]
fn version_zero_cannot_exist() {
    assert!(Version::new(0).is_err());
    assert_eq!(Version::new(1).unwrap().get(), 1);
}

#[test]
fn workspace_identity_binds_revision_and_digest() {
    let a = WorkspaceIdentity {
        revision: RevisionId::parse("rev1").unwrap(),
        digest: Digest::parse("hash1").unwrap(),
    };
    let b = WorkspaceIdentity {
        revision: RevisionId::parse("rev1").unwrap(),
        digest: Digest::parse("hash2").unwrap(),
    };
    assert_ne!(a, b);
}

#[test]
fn contract_identity_binds_version_and_environment() {
    let base = ContractIdentity {
        contract_id: ContractId::parse("contract:1").unwrap(),
        version: Version::new(1).unwrap(),
        digest: Digest::parse("sha256:test").unwrap(),
        toolchain: ToolchainId::parse("rust-stable").unwrap(),
        environment: EnvironmentId::parse("env:a").unwrap(),
    };
    let mut changed = base.clone();
    changed.environment = EnvironmentId::parse("env:b").unwrap();
    assert_ne!(base, changed);
}

#[test]
fn evidence_wire_parser_is_exact_and_fail_closed() {
    for (wire, expected) in [
        ("SUPPORTS", EvidenceRelation::Supports),
        ("REFUTES", EvidenceRelation::Refutes),
        ("UNRESOLVED", EvidenceRelation::Unresolved),
        ("OUT_OF_SCOPE", EvidenceRelation::OutOfScope),
    ] {
        assert_eq!(EvidenceRelation::try_from(wire).unwrap(), expected);
        assert_eq!(expected.as_wire(), wire);
    }
    for invalid in ["", "supports", "PASS", "UNKNOWN", "SUPPORTS "] {
        assert!(EvidenceRelation::try_from(invalid).is_err());
    }
}

#[test]
fn critical_states_are_distinct() {
    assert_ne!(RuntimeStatus::Active, RuntimeStatus::RecoveryRequired);
    assert_ne!(Verdict::DeliveryReady, Verdict::DeliveryBlocked);
    assert_ne!(ObligationStatus::Open, ObligationStatus::Closed);
}
