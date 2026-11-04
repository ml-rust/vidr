//! The `vidr-remaster` command. `vidr_remaster::main_with_args` holds its logic.

use std::process::ExitCode;

fn main() -> ExitCode {
    vidr_remaster::main_with_args(std::env::args_os())
}
