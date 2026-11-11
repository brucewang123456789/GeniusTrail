// Generates machine-readable protocol metadata from the one Rust source of truth.
fn main() {
    println!("{}", voder_protocol::contract_manifest().encode());
}
