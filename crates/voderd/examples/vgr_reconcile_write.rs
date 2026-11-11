//! Explicit recovery executable: operator must supply trusted state/session/run/action.
//! Refuses committed runs, divergent bytes, a missing WAL, or an active daemon.
use std::{env, path::Path};
use voderd::native_write_recovery::{reconcile_interrupted_write, Reconciliation};

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() != 5 {
        eprintln!("Usage: vgr_reconcile_write ABS_STATE_DIR SESSION_ID RUN_ID ACTION_ID");
        std::process::exit(2);
    }
    match reconcile_interrupted_write(Path::new(&args[1]), &args[2], &args[3], &args[4]) {
        Ok(Reconciliation::RolledBack) => println!("VGR_TRUSTED_WRITE_RECOVERY=ROLLED_BACK"),
        Ok(Reconciliation::AlreadyRestored) => {
            println!("VGR_TRUSTED_WRITE_RECOVERY=ALREADY_RESTORED")
        }
        Err(e) => {
            eprintln!("VGR_TRUSTED_WRITE_RECOVERY=REFUSED: {e}");
            std::process::exit(1);
        }
    }
}
