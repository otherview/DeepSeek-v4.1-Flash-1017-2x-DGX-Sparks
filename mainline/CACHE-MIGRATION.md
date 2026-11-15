# Native session-disk compatibility gate

The migration image is an experiment, not a replacement for the qualified Mia
spill-v2 deployment. Do not enable native persistent cache by merely mounting
the old spill directory or copying its flags.

At TensorFold `2ba55ec7`, `zig/src/families/deepseek_v41/sessions_gpu.zig`
opens its disk tier with **`fresh_start = true`**. The generic
`zig/src/sessions/disk_namespace.zig` then removes `.tfs` and `.tmp` files from
that exclusively locked compatibility/rank namespace during startup. The generic
disk tier supports other callers, but that does not establish persistent
recovery for this family. Changing the flag alone is not a qualified migration.

The native family exposes `TF_DSV41_SESSION_DISK`, `_DISK_GIB` and `_DISK_MIN`.
The inspected disk options and family environment parser do not expose equivalents
of the deployed 24-hour retention and 3.5 GiB read/write admission floor.
Startup memory floors and emergency container guards serve different purposes.

Required work before claiming cache parity:

1. Establish the family/store restart and checkpoint identity contract, including
   full snapshots, dependent deltas, partial writes and both-rank agreement.
2. Preserve 24-hour expiry for discovered and newly written snapshots, without
   leaving usable descendants whose base has expired or resetting age at boot.
3. Preserve read and write refusal below 3.5 GiB available memory, before staging
   allocations, with safe recomputation/skip behavior and rank agreement.
4. Exercise actual disk I/O with a controllable clock and memory probe: expired,
   boundary, fresh, malformed, missing base, interrupted write, low/high memory,
   restart and dependent-chain cases. Pure policy tests alone are insufficient.
5. Perform matched GPU fresh/warm and both-rank restart recovery trials in a new
   candidate namespace, with production spill data untouched.

Until implemented and tested, persistent native disk cache remains disabled in
the rendered experiment, and this is an explicit **production parity blocker**.
Offline AOT compilation does not resolve this gate or authorize a cutover.
