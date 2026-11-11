//! Stateless offline interop probe: one JSON value per line in, canonical JSON per line out.
use std::io::{self, BufRead};
fn main() {
    for input in io::stdin().lock().lines() {
        let Ok(input) = input else {
            std::process::exit(2);
        };
        match voder_protocol::json::parse(input.as_bytes()) {
            Ok(value) => println!("{}", value.encode()),
            Err(_) => println!("!REJECT!"),
        }
    }
}
