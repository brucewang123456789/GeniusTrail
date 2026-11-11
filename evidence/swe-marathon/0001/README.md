SWE-Marathon v1.1 official evaluation: metadata-only task receipt.

The model generation and evaluated submission are separate evidence objects.
The official reward belongs to the evaluated submission, not necessarily to
the model-generated artifact. The latter is not assigned this score.
Unknown provider identity, usage and submission hash remain null.
No solution, patch, answer, prompt, reasoning or raw evaluator log is present.

The base_commit field identifies the host Git baseline of the visible task snapshot supplied to Voder, including its workflow contract files. It is separate from official_source_commit.
The benchmark_source_url records the configured Git origin, including its download proxy.

Model/provider metadata is an operator configuration declaration read during metadata repair, not historical per-request model attestation. Runtime model ID and service endpoint remain unverified where null.

Metadata backfill uses same-run records and exact artifact hashes. Current configuration, where used, is a declaration rather than historical runtime proof. A submission hash identifies the artifact named by its explicit basis.

The evaluation-and-integrity duration is the complete official Harbor verifier wall interval. It includes integrity checks invoked inside that verifier, if present; it is not a separate integrity-only measurement.

The runtime-model-ID and API-service fields were filled by explicit operator declaration at metadata repair: deepseek-flash and https://api.deepseek.com. These values are not independently verified historical per-request observations.
