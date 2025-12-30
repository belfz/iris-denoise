mod cli;
mod img_io;
mod pipelines;
mod run;

use std::process;

use clap::Parser;

use crate::cli::Args;

fn main() {
    let args = Args::parse();
    if let Err(err) = run::run(args) {
        eprintln!("error: {err}");
        process::exit(1);
    }
}
