# Startup gate validation on vLLM 0.32

This independent candidate is based on PR17
`35f8f0569df4ffbed045bf0dc18974eaafd5ab1a`. Runtime changes are limited to
passing startup requirements and rejecting duplicate capability owners. Its
wheel contains 194 Python files and excludes the separate FP8/lifecycle changes.

Source/installed wheel candidate: `14b6f33c18b5b3796d3b9aa01408f06013924db2`.
Adapter wheel SHA256:
`616f19676caabf09b1bddc58a3a2e901db32f69c6c194947f032e7fe740e3649`.
Target: vLLM `a198423b7be0ccf9c9ac3fb01c02ac03ab145e93`.
Subsequent changes add only this report and the reproducible test-only CLI probe.

The source phase and independently installed phase each passed **66 CPU tests**:
`test_vllm_extensions.py`, `test_vllm_bridge.py`, and
`test_vllm_required_startup.py`. These are the same tests in two phases, not 132
different tests. The installed phase clears PYTHONPATH, runs outside the source
tree with importlib test import mode, and checks source/wheel/installed bytes.
Tests include a real fork, but injected platform/discovery dependencies do not
themselves prove real vLLM startup.

The fresh installed candidate was therefore also exercised through unmodified
official CLI `--help`, with actual platform/general-plugin metadata discovery
on gfx936/DTK. The small test-only distribution under
`tests/adapter/fixtures/required_capability_probe` supplies two extension specs;
their activation writes an event file and installs no operator. The observer
under `tests/adapter/tools/run_required_capability_cli_probe.py` runs the official
module and records its public `extension_reports()`, without replacing discovery,
callbacks or platform behavior.

| Installed CLI case | Result | Observed reports / activation events |
| --- | --- | --- |
| Unique capability owners, required `upgrade-probe-left` | Exit 0 | Both active; 2 events carry the requested capability |
| Two different owners and priorities advertise `upgrade-probe-left` | Exit 1 | Both failed with duplicate-owner reason; **0 events** |
| Unique owners, required `upgrade-probe-missing` | Exit 1 | Required-capability error; unrelated specs activate and record 2 events |

The missing case deliberately shows that required failure does not roll back
unrelated successful activation. This is an admission gate, not an arbitrary
extension transaction. CLI help runs before model/worker construction; no
real-model, Graph/AOT, precision or performance acceptance is claimed for this
independent wheel. The broader integration matrix remains open.

To reproduce, build/install this Adapter and the test-only fixture in a **fresh
isolated environment** with the target vLLM wheel. Keep PYTHONPATH empty and use
the recorded real HCU platform selection/toolchain paths. Set
`VLLM_PLUGINS=hcu_adapter`, `HCU_ADAPTER_HOST=none` to defer activation to the
official plugins. For each case use fresh absolute output/event paths:

```bash
python -m pip wheel --no-deps --no-build-isolation \
  tests/adapter/fixtures/required_capability_probe -w /path/to/probe-wheels
python -m pip install --no-deps /path/to/probe-wheels/hcu_required_capability_validation_probe-0.1.0-py3-none-any.whl
export HCU_ADAPTER_VLLM_REQUIRED_EXTENSIONS=
export HCU_ADAPTER_VLLM_REQUIRED_CAPABILITIES=upgrade-probe-left
export HCU_REQUIRED_PROBE_DUPLICATE=0
export HCU_REQUIRED_PROBE_EVENTS=/path/to/fresh-events.jsonl
python tests/adapter/tools/run_required_capability_cli_probe.py /path/to/fresh-report.json
```

For the duplicate case set `HCU_REQUIRED_PROBE_DUPLICATE=1`; for missing, use
unique owners and request `upgrade-probe-missing`. Both cases must exit 1;
inspect the JSON reports and events above. Do not install this test fixture in a
production/model environment.


## Actual Qwen3-8B pipeline model pair (R258)

Keep this request Draft. This adds healthy **real TP1/PP2 model** coverage;
refusal of an invalid owner before real PP transport remains unqualified.
The earlier native Worker/FULL Graph cases used runner/send stand-ins and
retain that separate scope. Normal lifecycle, performance, other required
hardware and independent approval remain open.

The matched installed pair uses Adapter main `8490a6b6072b` versus candidate
`6eeea95420cb`, with the same native sgl-kernel wheel, Core v0.33.0
`c440c089fe96`, merged Plugin v0.33.0 `c0e608de8153`, dependencies and
`/models/Qwen3-8B` checkpoint. The physical devices are gfx938/BW1100 GPUs6/7.
Both arms retain BF16 weights, E4M3 KV, FLASH_ATTN, block64, the default
FULL_AND_PIECEWISE compilation/Graph, prefix caching, max length8192,
max sequences8 and max batched tokens1024. Only Adapter Python payload changes.
No FlashAttention source or binary is changed or rebuilt.

| Observed result | Pure main | Candidate |
| --- | --- | --- |
| Real workers / distinct device UUIDs | 2 /2 | 2 /2 |
| PP0 /PP1 actual model layers, end-exclusive | 0–18 /18–36 | 0–18 /18–36 |
| Strict service probes /prefix-hit tokens | 17/17 /1280 | 17/17 /1280 |
| Actual HIP Graph launches per worker | 3 /3 | 3 /3 |
| Correlated GPU kernels per worker | 24 /24 | 24 /24 |
| HumanEval0–31 raw /isolated normalized | 20/32 /31/32 | 20/32 /31/32 |
| Completion tokens /length-truncated samples | 4186 /0 | 4174 /0 |
| Frontend exit0 within original15s /outer forced cleanup | Yes /none | Yes /none |
| Original complete lifecycle criterion | FAIL | FAIL |

All32 requests, finish reasons, raw pass flags and normalized pass flags match;
no newly failed task is observed. Messages and normalized completions match
28/32; tasks1/6/17/25 change. Usage matches29/32. The normalized failure remains
task26 in both arms. The original protocol does not retain generated token IDs;
these results do not establish output equivalence or token parity. Scores from
the earlier TP1 run are a different topology and are not a causal PP comparison.

Both arms report nanobind leaks, and Core internally sends SIGTERM to its two
workers after their cleanup grace. Main additionally logs EngineDeadError.
There are no surviving owned processes or outer-controller forced signals,
but exit0 and the15s bound do not waive these original lifecycle failures.
A profiler thread-initialization diagnostic is retained in both server logs;
the two actual device traces above were independently read and checked.

The worker observer uses each rank's current device/UUID and loaded layer span.
For main, `inspect.getsourcefile` initially identifies Torch's inference-mode
decorator; an additional installed-class `inspect.unwrap` observation resolves
Core's actual Worker method. Candidate resolves the thin Adapter Worker guard.
This static owner observation is distinguished from runtime worker metadata.
No owner state was corrupted and no fault-injection acceptance is claimed.

All18 owned distribution RECORDs are verified before/after each arm. Complete
payload comparison permits only hcu-adapter to change, normalizing exact console
shebang prefixes. Commands, environment, devices, requests, problems and protocol
match. The historical checkpoint manifest is checked by a fresh double read
before model evaluation and by another double read after each arm. Raw and
normalized generated-code scoring remains isolated under UID65534.

Reproduction uses the unchanged installed pair and the public CLI with
`--tensor-parallel-size 1 --pipeline-parallel-size 2
--distributed-executor-backend mp`. The recorded validation workspace contains
`evidence/adapter97-real-pp-pair-r258/run-pair.py`, both complete CLI/environment
receipts, the read-only worker observer, original outputs/traces, comparison
and audit scripts. This is local source-bound evidence, not a new remote CI run.

Audit: `evidence/adapter97-real-pp-audit-r258/audit.json`.
SHA256: `91a484091c4604b009c45c49c87d0fede9309d68a3ffb0953d7299e8d046d140`.
The complete347-row migration remains incomplete and last_good=false.
AI assistance: OpenAI Codex; this review does not supply independent approval.
