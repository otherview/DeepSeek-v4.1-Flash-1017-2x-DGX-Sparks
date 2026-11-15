# TensorFold main: opt-in two-Spark migration

This branch prepares an experimental native DeepSeek build on TensorFold main
`2ba55ec7d32d0bec446bb012678550be7ef52895` (the current main snapshot when prepared).
It does not replace the legacy recipe, change `scripts/local.sh` or `.env`, copy
anything to a Spark, start a model, modify compaction or download model weights.
The existing `start.sh`, `stop.sh`, `scripts/prepare.sh` and 53 legacy patches
remain untouched. Mainline is explicitly selected through `mainline/Makefile`.

## What is pinned and carried

`pins.json` freezes the TensorFold archive/checksum, compiler, build images,
reference tools, grammar library, and the exact heads and digests of two existing
upstream PRs:

- [#3026](https://github.com/ashhart/TensorFold/pull/3026), head `c9997e7c`: required/named and strict tool grammar integration, with its original authorship.
- [#3388](https://github.com/ashhart/TensorFold/pull/3388), head `baa16b1e`: complete workspace coverage for 18–32-row prefill segments, with its original authorship.

These are immutable snapshots, not moving PR refs. Their original commits and
contributors are recorded in the lock; we do not open competing TensorFold bug
fixes or claim those issues. The long-loop detector is already on main and is
not backported again. The candidate uses `--loop-guard` close-and-continue,
retains reasoning history and has no thinking budget or history-removal flag.

The known replay-boundary defect in [#2715](https://github.com/ashhart/TensorFold/issues/2715)
is not patched from a comment. This profile explicitly uses `TF_DSV41_PREFILL=full`.
This avoids that reported replay path; it is not proof that all other paths are correct.

`PATCHES.md` and `patch-reconciliation.json` account for every legacy patch and
all five deployed local changes. The old patch series is not applied to the new
engine. Owner mappings identify implementation and remaining gates; they do not
claim byte equivalence or silently retire a regression.

## Prepare and validate source on the workstation

```sh
make -C mainline prepare
python3 -m unittest discover -s mainline -p test_preparation.py
make -C mainline host-build ZIG=/path/to/pinned/linux-arm64-zig-directory
make -C mainline build-context
```

`prepare` verifies the pinned 36 MiB archive and both PR patches, applies them in
scratch, and refuses to overwrite a different existing source tree. The expanded
source is about 159 MiB. Source, build caches, binaries and logs stay under ignored
`mainline/data/`; a previously verified build context is reused, never replaced
silently. `host-build` uses a bounded CPU-only Linux ARM64 container (two CPUs,
3 GiB, runc, no network, GPU or model mounts). Compiler bytes must match the lock.

The CPU asset tools actually built, and the selected asset/token-map/draft/
forward host checks passed: **29/29 steps, 345 passed, 1 skipped**. Six migration
safety tests pass. See `VALIDATION.json`. These scoped checks are not the complete
TensorFold host/golden suite or real-model/device qualification.

## Prepared CUDA image build

The verified context is `mainline/data/context`. It contains only frozen source,
lock, Dockerfile and required notices; it contains no private configuration,
weights, SSH keys or captures. On a CPU build host with sufficient resources:

```sh
docker build --platform linux/arm64 \
  -t tensorfold-dsv41:mainline-2ba55ec7 mainline/data/context
```

The Dockerfile pins the NVIDIA base by digest, compiles the native CUDA head and
follower plus CPU asset tools, and builds `libtfgrammar` from the exact xgrammar
v0.2.8 commit/submodules. No model runs during image construction. Base images,
compiler/C++ caches and CUDA compilation need additional multi-GiB disk and
memory; the large NVIDIA base is already present on the production hosts, but
**no build was run there**. Do not compile on a busy serving host without a
separately authorized resource window.

The Linux ARM64 image **was built successfully on the workstation**, including
SM121 CUDA kernels, both native binaries and the grammar library. CPU-only smoke
checks passed: capabilities report CUDA/DeepSeek EXL3 and the planned thinking,
loop-guard, context, parallel and token-limit flags; server/asset help commands
run, and `libtfgrammar.so` loads. The real-library tool-grammar and budget suite also passed **52/52 tests**.
Image and binary hashes are in `VALIDATION.json`.
The initial build rejected a custom version override; the Dockerfile now uses
upstream version handling (reports 1.2.9), with the exact source identity carried
by the pinned revision and image labels. The image is local only, approximately
11.75 GB in Docker's reported size, with source/compiler caches included. No image
has been transferred to the Sparks. These checks do not qualify a real checkpoint,
GPU execution, two-node transport or performance.

## Model assets are a separate stage

Reuse the existing immutable EAQ-v1 checkpoint and matching overlay as read-only
input. Do not reuse the old executable, rank cache, token-map JSON, kit binaries
or spill-v2 files as if their formats were compatible. Mainline needs:

- `engram-token-map.bin` in DSENGR01 format, from its CPU token-map generator;
- `rope.bin` in DSV41RP1 format, with at least context + 2048 rows and 64 columns;
- AOT signatures/options/variants generated for **both** ranks, the exact SM121
  target, checkpoint geometry, draft settings, slot count and context;
- packed native Engram tables on each node and separate native prepared-weight
  cache directories.

The separately versioned reference is pinned in `pins.json`. Use its
`engine/zig/tools/zig/triton_fill.py`, `dsv41_triton_options.json` and
`dsv41_rope_tables.py` with `PYTHONPATH` pointing at its `engine/src`. Its compiler
contract names Triton 3.7.1 and ptxas 13.3.73. Run RoPE generation on the eventual
serving host CPU with the frozen build environment; preserve toolchain and file
provenance. Reference Python is a build-time aid, never a fallback inference engine.

The native CPU generator is invoked as:

```sh
tf-dsv41-token-map --tokenizer MODEL/tokenizer.json --config MODEL/config.json \
  --output ASSETS/engram-token-map.bin
tf-dsv41-assets aot-needs ASSETS/sigs.json ASSETS/needs-rank0.json
```

Run `aot-needs` under each rank's environment from `profile.py plan`, and compile
both needs sets through the frozen reference, retaining the union of variants.
Captured kernel images from a different checkpoint/profile are not interchangeable.
The reference packer/native Engram packer and native prepared-weight caches can
require another ~203 GB of packed Engram data per node and approximately a rank's
weights again. Inspect the actual plan and disk budget before creating them.
No conversion, Engram packing, model reads or large data copies were performed.

## Preserved-settings plan and explicit gaps

Copy `profile.example.json` to ignored `mainline/local.json` and supply existing
checkpoint and new asset/cache paths. Inspect without executing:

```sh
make -C mainline plan CONFIG=local.json
```

The plan preserves context 524288, pool 2621440, four slots, 32768 default reply
budget, thinking and guard-on behavior. It requests native RoCE transport and
RDMA plan links; interface selection and actual transport must be verified on
both nodes. Rank 1 starts before rank 0. Rank zero's API stays localhost:18036,
model `spark`; neither the routing gateway nor its settings are changed.

**The renderer never starts either command.** Its `activation_blockers` are real:

- The native disk store is different. The 24-hour expiry and 3.5 GiB read/write
  admission protections have no direct mapping yet. Persistent disk cache is
  therefore not mounted or enabled in the rendered commands. This is a blocking
  parity gap, not an accepted reduction of production behavior.
- Equivalent graph-cache bounds and actual memory headroom are unproven.
  Native startup floor settings do not replace the candidate-scoped 2.5 GiB
  soft / 1 GiB hard emergency guards, which must also be retained at activation.
- New assets, exact checkpoint/overlay load, drafts/plain, resumed/fresh,
  concurrent/solo, cancellation and guard continuation need GPU qualification.
- This initial candidate is text-only; vision remains unqualified and is not
  claimed equivalent to the legacy recipe.

These gaps prevent treating the plan as a production replacement. They do not
prevent preparing source, CPU generators and a native build context.

Before using generated assets, create `ASSETS/manifest.json` with the engine
revision, ordered patch digests, `profile_sha256` emitted by the plan, SHA256 of
checkpoint `config.json`/`tokenizer.json`, and a `files` map of every generated
asset and cubin to its SHA256. `verify-assets` checks those identities and basic
format/coverage; it does not establish CUDA launch coverage or numerical fidelity:

```sh
make -C mainline verify-assets CONFIG=local.json
```

Production remains on the already-qualified Mia deployment. No activation,
selection, routing, compaction, cache cleanup or rollback is performed here.
