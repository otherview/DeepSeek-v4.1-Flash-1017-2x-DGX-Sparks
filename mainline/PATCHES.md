# Patch reconciliation

Owner mapping is not proof of semantic or token equivalence. Every API/device/parity gate remains explicit.

The old patch directory and launcher remain intact for the legacy engine. The mainline build applies none of those 53 patches; it applies only the two immutable PR snapshots listed in `pins.json`.

| Legacy patch | Disposition | Owner / remaining work |
| --- | --- | --- |
| `0001-c7-indexer-split.patch` | new-owner | `zig/src/families/deepseek_v41/block_prefill_select.zig`: Native distributed indexer planning replaces the old patch point; remeasure TP2 prefill. |
| `0004-call-gate.patch` | replacement-pr | `zig/src/families/deepseek_v41/serve_engine.zig`: PR3026 wires required/named and strict tool grammars into the current core; libtfgrammar is required. |
| `0005-prefill-interleave.patch` | new-owner | `zig/src/families/deepseek_v41/sched_host.zig`: Current prompt-piece scheduler owns interleaving; full-prefill latency/cancellation must be tested. |
| `0008-pool-window.patch` | new-owner | `zig/src/families/deepseek_v41/cuda_kv.zig`: Use TF_DSV41_POOL_TOKENS and TF_DSV41_CONTEXT; assert resolved capacity, not just environment values. |
| `0010-rdma-idle-nap.patch` | revalidate | `zig/src/fabric/tp`: Current multi-rank transport replaces the old RDMA loop; idle power and wake-up behavior unmeasured. |
| `0011-cuda-budget-parked-pages.patch` | revalidate | `zig/src/families/deepseek_v41/cuda_kv.zig`: Current memory planner owns startup sizing; parked-page and actual headroom equivalence unproven. |
| `0012-dsv41-base-fixes.patch` | new-owner | `zig/src/families/deepseek_v41/engram_recovery_test.zig`: Old branch rebase fixes are not replayed; current Engram failure tests are the candidate owner. |
| `0013-core-spill-store.patch` | cache-migration | `zig/src/sessions`: Mainline session/disk store is a different implementation and format. No legacy spill data import. |
| `0014-core-cache-tier.patch` | cache-migration | `zig/src/families/deepseek_v41/sessions_gpu.zig`: Use native session adapter; prove continuation and concurrent restore behavior. |
| `0015-dsv41-kv-shard-attn.patch` | not-carried | `zig/src/families/deepseek_v41/cuda_kv.zig`: KV split is a different native path; no old KV-shard switch is forwarded or enabled for this profile. |
| `0016-dsv41-spill-blocks.patch` | cache-migration | `zig/src/families/deepseek_v41/sessions_gpu.zig`: Native session disk options replace TF_DS_SPILL_*; cache admission/expiry parity remains blocking. |
| `0017-dsv41-kit-row-split.patch` | new-assets | `zig/build/dsv41_images.zig`: Rebuild native family images and captured AOT variants for the pinned source; old kit binaries are not reused. |
| `0018-dsv41-kv-shard-lanes.patch` | not-carried | `zig/src/families/deepseek_v41/sessions_gpu.zig`: Do not transplant old sharded-lane code; qualify current multi-slot/page planner independently. |
| `0019-dsv41-spill-kv-shard.patch` | cache-migration | `zig/src/families/deepseek_v41/sessions_gpu.zig`: Old sharded spill format is incompatible; a new namespace and recovery tests are required. |
| `0020-dsv41-pair-broken.patch` | new-owner | `zig/src/fabric/tp`: Current transport/fate handling owns broken-pair termination; validate failure and cleanup on both nodes. |
| `0021-dsv41-spill-pictures.patch` | deferred-vision | `zig/src/families/deepseek_v41/sessions_gpu.zig`: Image-bearing cache states need separate qualification; this initial build profile is text-only. |
| `0022-dsv41-kv-shard-ring.patch` | not-carried | `zig/src/fabric/tp`: Use current native transport, never the old ring patch; compare transport correctness/performance. |
| `0023-server-prepare-memory.patch` | new-owner | `zig/src/server`: Current preparation, HTTP and memory admission paths replace old server structs; long-prompt peak RAM still needs measurement. |
| `0024-mixed-pack.patch` | revalidate | `zig/src/families/deepseek_v41/cuda_format.zig`: Mainline has its own EXL3/mixed-format planner; target EAQ v1 first, no new pack or overlay conversion. |
| `0025-widen-bf16.patch` | new-owner | `zig/src/families/deepseek_v41/plan.zig`: Native stored-dtype rules include F16 draft heads; exact derived checkpoint still requires a real load. |
| `0026-plain-rows.patch` | new-owner | `zig/src/families/deepseek_v41/draft/drive.zig`: Use native draft:false as the plain oracle; compare drafted/plain tokens on hardware. |
| `0027-tool-call-prose.patch` | api-gate | `zig/src/server`: Current server owns DSML/content separation; verify auto tools, content and finish reasons. |
| `0028-plain-request-text.patch` | api-gate | `zig/src/server`: Current tokenizer/prompt pipeline is different; control-token spellings require a request-level regression gate. |
| `0029-anthropic-budget-default.patch` | api-gate | `zig/src/server`: Recheck Anthropic thinking defaults on the current adapter instead of applying the older patch. |
| `0030-anthropic-context-management-lenient.patch` | api-gate | `zig/src/server`: Recheck context-management edits on the current Messages adapter. |
| `0031-anthropic-system-anywhere.patch` | api-gate | `zig/src/server`: Recheck system-message normalization without changing the client transcript. |
| `0032-stream-keepalive-ping.patch` | api-gate | `zig/src/server`: Verify SSE keepalive and prefill cancellation through the native server and existing proxy. |
| `0033-graph-cap.patch` | blocking-parity | `zig/src/families/deepseek_v41/round_graph.zig`: The old GRAPH_CAP=128 has no verified one-to-one mapping to current round graphs; bound and measure memory. |
| `0034-rebase-followups.patch` | branch-specific | `zig/src/families/deepseek_v41/index_scratch_test.zig`: Do not replay old rebase-only offsets; current plan/alignment tests and PR3388 cover the new owner. |
| `0035-engram-lead-failure-paths.patch` | new-owner | `zig/src/families/deepseek_v41/engram_recovery_test.zig`: Current Engram failure/recovery and allocation sweeps own cleanup; validate actual host I/O failures later. |
| `0036-metrics-cached-prompt-tokens.patch` | api-gate | `zig/src/families/deepseek_v41/serve_engine.zig`: Validate cached prompt counters, timings and normalized discovery against our proxy/client expectations. |
| `0037-rendezvous-ports-early.patch` | revalidate | `zig/src/fabric/tp`: Current rendezvous lifecycle replaces old early port opens; verify worker-before-head startup and bounded failure. |
| `0038-aligned-index-views.patch` | new-owner | `zig/src/families/deepseek_v41/index_scratch_test.zig`: New buffer planner has explicit coverage tests; carry PR3388 and validate emitted GPU views. |
| `0040-restore-room.patch` | cache-migration | `zig/src/families/deepseek_v41/sessions_gpu.zig`: No claim that native page/session admission preserves the old prefix under pressure; test that case explicitly. |
| `0041-service-between-pieces.patch` | new-owner | `zig/src/core/lanes`: Current lane host owns queued/parked service; validate cancellation and fair progress under long prefill. |
| `0042-kv-shard-pieces.patch` | not-carried | `zig/src/families/deepseek_v41/sched_host.zig`: Old sharded prompt-piece integration is not applied; profile uses native unsharded pool and prompt pieces. |
| `0043-picture-resume-skip.patch` | deferred-vision | `zig/src/families/deepseek_v41/sessions_gpu.zig`: Picture-prefix reuse is outside the initial text-only profile and remains unqualified. |
| `0045-cut-in-tail.patch` | revalidate | `zig/src/families/deepseek_v41/sched_host.zig`: Current scheduler/retained-prefix accounting needs a cold/resumed comparison; no byte-equivalence claim. |
| `0046-server-get-delete-body-discard.patch` | api-gate | `zig/src/server`: Current HTTP framing/body bounds own request disposal; verify pipelined reuse without old struct patches. |
| `0047-replayed-reasoning.patch` | api-gate | `zig/src/server`: Retain reasoning history; verify reasoning/ reasoning_content aliases through actual prompt normalization. |
| `0048-stream-tool-calls.patch` | api-gate | `zig/src/families/deepseek_v41/serve/tool_stream.zig`: Use current DSML streaming with PR3026; verify required/named/auto and split chunks. |
| `0049-first-draft-depth.patch` | new-owner | `zig/src/families/deepseek_v41/draft/depth.zig`: Current DSpark depth policy replaces the old first-draft-depth patch; verify plain equality. |
| `0050-draft-calibration.patch` | new-owner | `zig/src/families/deepseek_v41/draft/calib.zig`: Native cost calibration is a different subsystem; legacy draft calibration weight is not forwarded. |
| `0051-prompt-rows.patch` | new-owner | `zig/src/families/deepseek_v41/serve_knobs.zig`: Current prefill-row/chunk settings replace old prompt-rows controls; record resolved settings and benchmark. |
| `0056-server-picture-prepare-scratch.patch` | deferred-vision | `zig/src/server`: No picture preparation/vision claim until independent model and memory qualification. |
| `0057-drop-reasoning-history.patch` | policy-retained | `zig/src/server`: Do not pass history-removal flags or mutate client messages; current rendered-history behavior must pass its API gate. |
| `0058-responses-include-encrypted.patch` | api-gate | `zig/src/server`: Recheck Responses include=reasoning.encrypted_content behavior; no legacy parser patch applied. |
| `0059-lane-thread-binds-context.patch` | new-owner | `zig/src/families/deepseek_v41/serve_engine.zig`: Current lane host binds the CUDA context via on_thread; host build covers type integration, device test still needed. |
| `0064-dsml-opener-short-payload.patch` | api-gate | `zig/src/families/deepseek_v41/serve/tool_scan.zig`: Current DSML scanner replaces old parser; short/truncated-block regression must pass. |
| `0065-dsml-value-holds-closer.patch` | api-gate | `zig/src/families/deepseek_v41/serve/tool_scan.zig`: Value-containing-closer cases require current parser goldens; not assumed fixed by a filename match. |
| `0066-load-prefetch.patch` | new-owner | `zig/src/families/deepseek_v41/cuda_load_prefetch.zig`: Use current loader and a new prepared cache; never treat legacy rank-cache bytes as native prepared images. |
| `0067-orphan-dsml-closer-diagnostic.patch` | api-gate | `zig/src/families/deepseek_v41/serve/tool_scan.zig`: Verify orphan DSML closer diagnostics and empty-call handling with real streamed/buffered requests. |
| `0068-rdma-proxy-cpus.patch` | revalidate | `zig/src/fabric/tp`: Old TF_DS_RDMA_CPUS is not forwarded; native transport affinity must be selected from its own supported settings. |

## Deployed local fixes

- **long-loop guard — already upstream:** Use --loop-guard; do not apply our old backport.
- **first-token/opener correction — different implementation:** Use current masked first-token path and verify required/named tool cases.
- **long-prefix preservation — parity gate:** Repeat retained-prefix-under-pressure reproduction against native pool.
- **cache TTL and memory floor — blocking parity gap:** No mapping to legacy spill-v2; port/test native policy before enabling persistent disk cache.
- **reasoning-history regression tests — policy retained, API gate pending:** Retain all reasoning history and verify rendered prompts with current server.
