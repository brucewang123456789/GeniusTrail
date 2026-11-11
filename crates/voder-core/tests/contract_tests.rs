use voder_core::contract::*;
use voder_core::types::*;

fn task() -> TaskAnchor {
    TaskAnchor {
        task_id: TaskId::parse("task-1").unwrap(),
        identity: "fixed issue #1".into(),
        goal: "Fix test without regression".into(),
        version: Version::new(1).unwrap(),
        accepted_facts: vec!["Repository revision pinned".into()],
        environment_constraints: vec!["Linux".into()],
        evaluation_boundary: vec!["No production deployments".into()],
    }
}
fn draft() -> ContractDraft {
    let rid = RequirementId::parse("tests-pass").unwrap();
    ContractDraft {
        anchor: task(),
        identity: ContractIdentity {
            contract_id: ContractId::parse("contract-1").unwrap(),
            version: Version::new(1).unwrap(),
            digest: Digest::parse("sha256:contract").unwrap(),
            toolchain: ToolchainId::parse("rust-1").unwrap(),
            environment: EnvironmentId::parse("linux-fingerprint").unwrap(),
        },
        requirements: vec![Requirement {
            id: rid.clone(),
            description: "Relevant tests pass".into(),
            hard: true,
            severity: Severity::Critical,
        }],
        validators: vec![ValidatorBinding {
            id: ValidatorId::parse("cargo-test").unwrap(),
            version: ValidatorVersion::parse("1").unwrap(),
            requirement_ids: vec![rid],
        }],
        trusted_sources: vec![SourceId::parse("cargo-test").unwrap()],
        tools: vec![ToolRule {
            id: ToolId::parse("patch").unwrap(),
            effect: ToolEffect::Write,
            requires_approval: true,
        }],
        read_scopes: vec![
            RepoScope::parse("src").unwrap(),
            RepoScope::parse("tests").unwrap(),
        ],
        write_scopes: vec![RepoScope::parse("src").unwrap()],
        forbidden_effects: vec!["secret_exfiltration".into()],
        required_inputs: vec![RequiredInput {
            id: InputId::parse("project-file").unwrap(),
            kind: InputKind::File,
            description: "Original source file".into(),
            why_required: "Needed to locate defect".into(),
        }],
        sampling_rules: vec![SamplingRule {
            tool_id: ToolId::parse("patch").unwrap(),
            obligation_id: ObligationId::parse("diagnosis-1").unwrap(),
            max_samples: 2,
        }],
    }
}

#[test]
fn contract_compiles_and_has_no_setters() {
    let c = FrozenContract::compile(draft()).unwrap();
    assert_eq!(c.identity().version.get(), 1);
    assert_eq!(c.anchor().goal, "Fix test without regression");
    assert_eq!(c.validators().len(), 1);
    assert_eq!(c.sampling_rules()[0].max_samples, 2);
    assert!(c.read_scopes()[0].contains(&RepoScope::parse("src/lib.rs").unwrap()));
    assert!(!c.write_scopes()[0].contains(&RepoScope::parse("tests/check.rs").unwrap()));
}

#[test]
fn missing_inputs_are_explicit_and_do_not_turn_into_success() {
    let c = FrozenContract::compile(draft()).unwrap();
    let missing = c.missing_inputs(&[]);
    assert_eq!(missing.len(), 1);
    assert_eq!(missing[0].description, "Original source file");
    assert_eq!(missing[0].kind, InputKind::File);
    assert_eq!(missing[0].why_required, "Needed to locate defect");
    assert!(c
        .missing_inputs(&[InputId::parse("project-file").unwrap()])
        .is_empty());
    assert_eq!(
        c.missing_inputs(&[InputId::parse("unrelated").unwrap()])
            .len(),
        1
    );
}

#[test]
fn no_hard_requirement_is_rejected() {
    let mut d = draft();
    d.requirements[0].hard = false;
    assert!(FrozenContract::compile(d)
        .unwrap_err()
        .contains(&ContractError::MissingHardRequirement));
}

#[test]
fn absent_or_wrong_validator_mapping_cannot_certify_hard_requirement() {
    let mut d = draft();
    d.validators[0].requirement_ids.clear();
    assert!(FrozenContract::compile(d)
        .unwrap_err()
        .contains(&ContractError::HardRequirementWithoutValidator));
    let mut d = draft();
    d.validators[0].requirement_ids = vec![RequirementId::parse("different").unwrap()];
    let errors = FrozenContract::compile(d).unwrap_err();
    assert!(errors.contains(&ContractError::UnknownRequirement));
    assert!(errors.contains(&ContractError::HardRequirementWithoutValidator));
}

#[test]
fn empty_sources_or_validators_fail_closed() {
    let mut d = draft();
    d.trusted_sources.clear();
    assert!(FrozenContract::compile(d)
        .unwrap_err()
        .contains(&ContractError::MissingValidationRoots));
    let mut d = draft();
    d.validators.clear();
    assert!(FrozenContract::compile(d)
        .unwrap_err()
        .contains(&ContractError::MissingValidationRoots));
}

#[test]
fn duplicate_requirement_tool_validator_source_input_and_policy_fail() {
    let mut d = draft();
    d.requirements.push(d.requirements[0].clone());
    assert!(FrozenContract::compile(d)
        .unwrap_err()
        .contains(&ContractError::Duplicate("requirement")));
    let mut d = draft();
    d.tools.push(d.tools[0].clone());
    assert!(FrozenContract::compile(d)
        .unwrap_err()
        .contains(&ContractError::Duplicate("tool")));
    let mut d = draft();
    d.validators.push(d.validators[0].clone());
    assert!(FrozenContract::compile(d)
        .unwrap_err()
        .contains(&ContractError::Duplicate("validator")));
    let mut d = draft();
    d.trusted_sources.push(d.trusted_sources[0].clone());
    assert!(FrozenContract::compile(d)
        .unwrap_err()
        .contains(&ContractError::Duplicate("source")));
    let mut d = draft();
    d.required_inputs.push(d.required_inputs[0].clone());
    assert!(FrozenContract::compile(d)
        .unwrap_err()
        .contains(&ContractError::Duplicate("required_input")));
    let mut d = draft();
    d.sampling_rules.push(d.sampling_rules[0].clone());
    assert!(FrozenContract::compile(d)
        .unwrap_err()
        .contains(&ContractError::Duplicate("sampling_policy")));
}

#[test]
fn write_scope_must_be_covered_by_read_scope() {
    let mut d = draft();
    d.write_scopes = vec![RepoScope::parse("private").unwrap()];
    assert!(FrozenContract::compile(d)
        .unwrap_err()
        .contains(&ContractError::WriteScopeOutsideReadScope));
    let mut d = draft();
    d.write_scopes.clear();
    assert!(FrozenContract::compile(d)
        .unwrap_err()
        .contains(&ContractError::WriteToolWithoutScope));
}

#[test]
fn read_only_contract_requires_no_write_scope() {
    let mut d = draft();
    d.tools[0].effect = ToolEffect::Read;
    d.write_scopes.clear();
    assert!(FrozenContract::compile(d).is_ok());
}

#[test]
fn invalid_sampling_policy_cannot_authorize_repetition() {
    let mut d = draft();
    d.sampling_rules[0].max_samples = 0;
    assert!(FrozenContract::compile(d)
        .unwrap_err()
        .contains(&ContractError::InvalidSamplingPolicy));
    let mut d = draft();
    d.sampling_rules[0].tool_id = ToolId::parse("unregistered").unwrap();
    assert!(FrozenContract::compile(d)
        .unwrap_err()
        .contains(&ContractError::InvalidSamplingPolicy));
}

#[test]
fn path_lexical_gates_reject_traversal_ambiguity_and_windows_devices() {
    for name in [
        "",
        "/etc",
        "~/.ssh",
        "src/../secret",
        "src/./file",
        "src//file",
        "src\\file",
        "C:/root",
        "src/file:",
        "src/file\nname",
        "src/CON",
        "aux.txt",
        "src/com1.log",
        "src/name.",
        "src/name ",
    ] {
        assert_eq!(
            RepoScope::parse(name),
            Err(ContractError::InvalidScope),
            "{name:?}"
        );
    }
    assert!(RepoScope::parse("src/app.rs").is_ok());
    let src = RepoScope::parse("src").unwrap();
    assert!(src.contains(&RepoScope::parse("src/sub/a.rs").unwrap()));
    assert!(!src.contains(&RepoScope::parse("src-extra/file").unwrap()));
}

#[test]
fn empty_goal_and_invalid_requirements_are_rejected() {
    let mut d = draft();
    d.anchor.goal = "  ".into();
    assert!(FrozenContract::compile(d)
        .unwrap_err()
        .contains(&ContractError::Empty("goal")));
    let mut d = draft();
    d.required_inputs[0].why_required = "".into();
    assert!(FrozenContract::compile(d)
        .unwrap_err()
        .contains(&ContractError::Empty("input_description_or_reason")));
}

#[test]
fn anchor_and_compiled_identity_survive_as_independent_domains() {
    let c = FrozenContract::compile(draft()).unwrap();
    assert_eq!(c.anchor().version.get(), 1);
    assert_eq!(c.identity().version.get(), 1);
    assert_ne!(
        c.anchor().task_id.as_str(),
        c.identity().contract_id.as_str()
    );
}
