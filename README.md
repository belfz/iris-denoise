# denoise

Rust CLI to denoise astrophotography frames (TIFF, PNG, JPEG) using a median filter plus a light Gaussian blur. Includes a gentle strength curve and parallelized median filtering via Rayon.

## Requirements
- Rust toolchain (stable recommended)

## Build
- Debug (fast compile): `cargo build`
- Release (optimized): `cargo build --release`

## Run
```
cargo run --release -- <input.png> [output.png] [--strength 1-5]

# or after building:
./target/release/denoise -h
```

## Notes
- Median is parallelized; Rayon uses all cores by default. Limit threads via `RAYON_NUM_THREADS=4 ./target/release/denoise ...` if desired.
- Strength mapping (kernel, sigma): 1→(1,0.10), 2→(2,0.18), 3→(3,0.30), 4→(4,0.45), 5→(5,0.60).

