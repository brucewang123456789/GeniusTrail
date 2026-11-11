use voder_core::{budget::*, contract::*, types::*};

/// Missing prerequisites are surfaced before any resource is reserved. This is
/// the first two-gate building block, NOT a full orchestrator or evidence gate.
#[test]
fn missing_file_blocks_costly_work_until_trusted_inventory_confirms_it() {
    let requirement = RequirementId::parse("hard-pass").unwrap();
    let file_id = InputId::parse("project").unwrap();
    let draft = ContractDraft {
        anchor: TaskAnchor {
            task_id: TaskId::parse("t").unwrap(),
            identity: "t@v1".into(),
            goal: "Fix reported bug".into(),
            version: Version::new(1).unwrap(),
            accepted_facts: vec![],
            environment_constraints: vec![],
            evaluation_boundary: vec![],
        },
        identity: ContractIdentity {
            contract_id: ContractId::parse("c").unwrap(),
            version: Version::new(1).unwrap(),
            digest: Digest::parse("d").unwrap(),
            toolchain: ToolchainId::parse("rust").unwrap(),
            environment: EnvironmentId::parse("linux").unwrap(),
        },
        requirements: vec![Requirement {
            id: requirement.clone(),
            hard: true,
            severity: Severity::High,
            description: "Test passes".into(),
        }],
        validators: vec![ValidatorBinding {
            id: ValidatorId::parse("test").unwrap(),
            version: ValidatorVersion::parse("1").unwrap(),
            requirement_ids: vec![requirement],
        }],
        trusted_sources: vec![SourceId::parse("test").unwrap()],
        tools: vec![ToolRule {
            id: ToolId::parse("read").unwrap(),
            effect: ToolEffect::Read,
            requires_approval: false,
        }],
        read_scopes: vec![RepoScope::parse("src").unwrap()],
        write_scopes: vec![],
        forbidden_effects: vec![],
        required_inputs: vec![RequiredInput {
            id: file_id.clone(),
            kind: InputKind::File,
            description: "Upload src/lib.rs".into(),
            why_required: "Source is absent".into(),
        }],
        sampling_rules: vec![],
    };
    let contract = FrozenContract::compile(draft).unwrap();
    let mut budget = BudgetLedger::new(BudgetAmount {
        steps: 2,
        model_calls: 2,
        tokens: 500,
        ..BudgetAmount::default()
    });
    let budget_before = budget.snapshot();
    assert_eq!(contract.missing_inputs(&[]).len(), 1);
    assert_eq!(
        budget.snapshot(),
        budget_before,
        "must not charge while waiting for file"
    );
    assert!(contract.missing_inputs(&[file_id]).is_empty());
    let reservation_id = ReservationId::parse("api-request-1").unwrap();
    assert_eq!(
        budget
            .reserve(
                reservation_id,
                BudgetAmount {
                    steps: 1,
                    model_calls: 1,
                    tokens: 250,
                    ..BudgetAmount::default()
                }
            )
            .unwrap(),
        ReserveOutcome::Created
    );
}
