# iris-denoise

Rust CLI to denoise astrophotography frames (TIFF, PNG, JPEG) using a denoise algorithm of choice.

## Requirements
- Rust toolchain (stable recommended)

## Build
- Debug (fast compile): `cargo build`
- Release (optimized): `cargo build --release`

## Run
```
cargo run --release -- <input.png> [output.png] [--strength 1-5]

# or after building:
./target/release/iris-denoise -h
```
