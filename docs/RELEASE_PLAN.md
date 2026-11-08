# Build and release plan

## Stage 0 — Repository foundation (current)

No AI model or native library exists yet.

GitHub Actions should:
- verify the README and project constraints;
- run Python syntax checks when Python source files are added;
- compile-check C++ source files when they are added;
- avoid publishing placeholder binaries, random model files, or unverified performance claims.

Expected output: CI pass/fail status only. No downloadable release artifact is needed yet.

## Stage 1 — Tiny learning prototype

Once the first minimal model and tests exist, CI may upload:
- test results;
- benchmark report (RAM usage, CPU time, model size, and test configuration);
- optional debug build, clearly labeled as a development artifact.

These are temporary workflow artifacts, not official releases.

## Stage 2 — First usable source release

After APIs and tests stabilize, a tagged release may include:
- source code (GitHub also provides source archives for tags);
- Python package or source distribution, if packaging is ready;
- build instructions and supported platform information;
- benchmark results and SHA-256 checksums.

## Stage 3 — Native library releases

Only after the C++ library builds and tests on a specific platform, attach platform-specific binaries, for example:
- Linux x86_64: shared library such as `libsimple_ai.so`;
- Android ARM64: Android-compatible library only after NDK/JNI integration and device testing;
- other platforms only when a reproducible build is available.

Do not bundle one platform's binary and imply it works on all devices.

## Stage 4 — Model checkpoint releases

Model weights/checkpoints may be released only when:
- they were actually trained by this project;
- training data and data-processing provenance are documented;
- license and redistribution status are clear;
- model format and loading code are documented;
- file size and measured RAM/CPU requirements are reported;
- offline operation has been tested.

A checkpoint is not automatically included in every source release. Large model files may need a separate release asset.

## Release artifact naming (future)

Use descriptive, versioned names, such as:
- `simple-ai-python-vX.Y.Z.tar.gz`
- `simple-ai-linux-x86_64-vX.Y.Z.tar.gz`
- `simple-ai-android-arm64-vX.Y.Z.tar.gz`
- `simple-ai-checkpoint-vX.Y.Z.bin`
- `simple-ai-benchmarks-vX.Y.Z.zip`

These are examples only—not files that exist today.

## Resource validation

The project target is **at most 1.2 GB RAM and 1 GHz CPU for both training and inference**. Each release should report the actual environment and measured peak RAM/CPU behavior. CI runners are not equivalent to the target Android/Termux device, so CI results alone cannot prove the target constraint is met.
