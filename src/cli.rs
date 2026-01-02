use std::path::PathBuf;

use clap::{Parser, ValueEnum};

#[derive(ValueEnum, Debug, Copy, Clone, Eq, PartialEq)]
pub enum Strategy {
    Default,
    Experimental,
    ATrous,
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
pub struct Args {
    /// Input image path
    #[arg(value_name = "INPUT")]
    pub input: PathBuf,

    /// Output image path (defaults to auto-named when omitted)
    #[arg(short, long, value_name = "OUTPUT")]
    pub output: Option<PathBuf>,

    /// Denoise strength 1-5 (mild to strong). Only applies to Default and Experimental models. If not specified, the default strength is 3.
    #[arg(
        short,
        long,
        value_name = "STRENGTH",
        default_value_t = 3,
        value_parser = clap::value_parser!(u8).range(1..=5)
    )]
    pub strength: u8,

    /// Enable sharpening; optional strength 1-5, defaults to 3 when flag is given without a value
    #[arg(
        long,
        value_name = "STRENGTH",
        num_args = 0..=1,
        default_missing_value = "3",
        value_parser = clap::value_parser!(u8).range(1..=5)
    )]
    pub sharpen: Option<u8>,

    /// Strategy (algorithm) to use for denoising. Choose between Default, Experimental, and ATrous.
    #[arg(short, long, value_parser = clap::value_parser!(Strategy))]
    pub strategy: Strategy,
}
