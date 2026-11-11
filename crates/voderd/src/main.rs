//! Thin stdio entry. Never emits diagnostics on protocol stdout.
use std::{env, io, path::PathBuf};
fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() != 3 || args[1] != "--state-dir" {
        eprintln!("Usage: voderd --state-dir /absolute/private/state/dir");
        std::process::exit(2);
    }
    let state = PathBuf::from(&args[2]);
    if !state.is_absolute() {
        eprintln!("state dir must be absolute");
        std::process::exit(2);
    }
    match voderd::protocol_stdio::Server::open(&state)
        .and_then(|mut s| s.serve(io::stdin().lock(), io::stdout().lock()))
    {
        Ok(()) => {}
        Err(e) => {
            eprintln!("voderd stopped (fail closed): {e}");
            std::process::exit(1);
        }
    }
}
