# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Environment choices for official SGLang startup, resolved one setting at a time."""

def automatic_values(args, family, env):
    """Inherit deployment inputs after explicit and file settings are combined."""
    auto = {}
    if family is None:
        return auto
    ep = args.moe_a2a_backend == "deepep" and args.ep_size != "1"
    pd = args.disaggregation_transfer_backend == "mooncake"
    decode = args.disaggregation_mode == "decode"
    fp8 = family == "fp8"
    w4 = family == "w4a8"
    hca = env.get("NCCL_IB_HCA")
    if "NCCL_SOCKET_IFNAME" in env:
        auto["GLOO_SOCKET_IFNAME"] = env["NCCL_SOCKET_IFNAME"]
    if ep and hca:
        auto["ROCSHMEM_ALLOWED_IBV_DEVICES"] = hca
    if pd and args.disaggregation_ib_device and (decode or w4):
        auto["MC_TE_FILTERS"] = args.disaggregation_ib_device
        if fp8:
            auto["MC_ALLOWED_IBV_DEVICES"] = args.disaggregation_ib_device
    if w4 and "NO_PROXY" in env:
        auto["no_proxy"] = env["NO_PROXY"]
    return auto


def resolve_environment(args, family, inherited, configured, keys):
    # Rules that depend on site inputs see the same explicit/config precedence.
    auto = automatic_values(args, family, {**configured, **inherited})
    # Preserve the channel-INT4 checkpoint contract even with a custom YAML.
    defaults = {"HCU_W4A8_MOE": "native"} if family == "w4a8" else {}
    env = inherited.copy()
    sources = {}
    for key in keys:
        for source, values in (("explicit", inherited), ("config", configured),
                               ("auto", auto), ("default", defaults)):
            if key in values:
                env[key] = str(values[key])
                sources[key] = source
                break
        # Otherwise leave the owning runtime's default in effect.
    return env, sources
