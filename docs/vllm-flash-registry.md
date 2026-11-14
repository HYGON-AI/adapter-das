# Flash publication and process-local health

The transaction implementation is shared in `hcu_adapter/runtime_lifecycle.py`
(stdlib plus framework-free extension protocol). The vLLM bridge supplies only
its protected runtime action ID and existing public error type, public wrappers,
admission and selector/activation orchestration. Locks, report/PID bindings,
reentry, fork and failure-before-decrement remain one transaction in the shared
module. It imports neither framework nor Torch/kernel/provider package.
SGLang receives no new activation policy from this extraction. Shared contracts
exercise independent named consumers; both actual Adapter host dispatch/import
paths regress alongside the actual vLLM selector and metadata/spawn suites.

vLLM owns attention selection and configuration validation. Adapter requires a
unique ACTIVE `hcu-flash-attention` report and its exact current-process health
binding, then delegates to `RocmPlatform`. No Plugin class path or backend table
is maintained here. Default Triton and official MLA/Sparse routing remain intact.

`ExtensionSpec.check_health` is an optional final API1 field. Before activation,
it must be synchronous and `inspect.signature(checker).bind()` must succeed.
Optional parameters are allowed. A checker returns None and must not re-enter
host lifecycle APIs. Its binding uses the exact ACTIVE report object and PID;
replacement, duplicate and failed reports cannot borrow an earlier binding.
Failure in the post-activation binding sink is commit-fatal, with one FAILED
report for that owner.
Diagnostic rendering in the sink-failure path also uses PR23's guarded renderer;
a broken `__str__` cannot replace the typed fatal outcome or its original cause.

Lock order is host snapshot, unlocked owner callback, host verification. The
inflight counter covers all validation, callback, verification and failure
publication. A fixed failure sentinel is stored before diagnostic formatting;
formatting failure cannot erase refusal. Fork clears successful reports and
bindings, while incomplete or failed runtime/health state remains refused.
All selector routes check the runtime latch. Public registry writes have no
CAS: these checks do not promise atomicity against arbitrary concurrent writers.

At the vLLM public `HcuPlatform.set_additional_forward_context` boundary,
`require_forward_healthy()` preserves those built-in checks and calls the shared
`require_requested_extensions_healthy()` transaction before delegating arguments
and the result unchanged to `RocmPlatform`. After successful extension activation
and report publication, the runtime callback calls
`record_forward_extension_policy(context)` without running any checker. It freezes
the requested entry-point names and capabilities, their exact ACTIVE owner union,
original checker bindings (or explicit API1 no-checker markers), PID and report
generation. Each selected owner is checked once, outside the host lock; unrequested
optional owners are outside this group. Names refer to `entry_point_name`, not the
extension's declared name. Capability ownership retains the strict unique-ACTIVE
selection rule. An enrolled `check_health=None` owner remains compatible, while the
separate strict `require_extension_healthy(capability)` still requires a checker.

Forward requires completed current-PID runtime activation and trusted enrollment,
even with no requested owners. Preactivation, incomplete activation and cleared
healthy-fork enrollment refuse; the normal child runtime plugin must activate
first. Failed/inflight fork state remains refused. `require_runtime_healthy()`
continues to read the failure latch only, without proving activation completed.
Requests are frozen at activation, so forward neither rereads environment policy
nor activates or repairs owners. Every later report publication invalidates the
forward policy, including republication of the same report objects. Missing or
replaced bindings cannot become no-checker markers or be revived by remove/readd.
The existing strict capability API retains its separate publication semantics.

CPU lifecycle, public-hook ordering and bootstrap regressions are source evidence.
Genuine loaded compiled-model execution and live Graph replay through the official
runner, healthy then after controlled owner drift in disposable workers, remain
separate open acceptance gates; Python fixtures do not establish those results.

The host also checks each built-in mHC and sparse-indexer owner separately.
Importing both `require_healthy` callables into one local name loses the first
check; use distinct bindings. Regress both owner failures and the all-healthy
path (`tests/adapter/test_vllm_builtin_health.py`), plus real installed public
constructor drift and restored-policy refusal. A healthy indexer cannot attest
to the mHC owner's registration or process health.

The Plugin prepares predictable dependencies before public FLASH_ATTN
registration, refuses foreign overrides, and exposes its own health callback.
Registry drift must fail through the actual official selector, even if the
Plugin class would no longer be selected. No RMS, W4, FLA or model-load hook is
required by this independent owner.

Dependencies are published PR21 plus corrected PR23 (both based on PR17), with
the paired Plugin PR326 public-registry delta. CPU acceptance covers extension
ABI/lifecycle, bounded actual fork, and the actual target public selector for
clear/foreign drift, configuration refusal, optional/required failure, spawn and
unpickle. CPU evidence is not a numerical or full-model claim. Run the focused
tests in `tests/adapter/test_vllm_extension_health.py`,
`test_vllm_commit_fatal.py`, `test_vllm_flash_capability.py`, and the paired
Plugin `tests/vllm032/test_flash_public_selector.py` in isolated environments.
The inherited PR23 metadata-loader fixture deliberately rejects unrelated
installed extension entry points. Run it in a separate Adapter+target venv
without the production Plugin; do not filter discovery to simulate that state.
Its required-context CPU caller must forward the additive health sink keyword.
Fresh-interpreter checks retain their original health assertions and timeout.
