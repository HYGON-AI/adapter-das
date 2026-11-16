import pytest
import torch
from hcu_adapter.fla_hcu.chunk_delta_h import chunk_gated_delta_rule_fwd_h


@pytest.mark.parametrize('dim', [64, 128])
@pytest.mark.parametrize('dtype', [torch.bfloat16, torch.float32])
def test_snapshot_matches_prefix_and_preserves_untracked(dim, dtype):
    torch.manual_seed(19)
    dev='cuda'; heads=2; lens=[193,70,129]; total=sum(lens)
    k=torch.randn(1,total,heads,dim,device=dev,dtype=torch.bfloat16)*0.01
    u=torch.randn_like(k)*0.1
    w=torch.zeros_like(k)
    backing=torch.randn(5,2,heads,dim,dim,device=dev,dtype=dtype)*0.1
    pool=backing[:,1]
    original=pool.clone()
    snapshot=torch.full((3,heads,dim,dim),float('nan'),device=dev)
    indices=torch.tensor([3,0,2],device=dev,dtype=torch.int32)
    cu=torch.tensor([0,193,263,total],device=dev,dtype=torch.int32)
    track=torch.tensor([2,0,-1],device=dev,dtype=torch.int32)
    h,_=chunk_gated_delta_rule_fwd_h(k,w,u,initial_state=pool,initial_state_indices=indices,cu_seqlens=cu,track_state=snapshot,track_chunk_idx=track)
    torch.testing.assert_close(snapshot[1],original[0].float(),rtol=0,atol=0)
    assert torch.isnan(snapshot[2]).all()
    assert torch.isfinite(snapshot[:2]).all()
    expected=original[3].float()+torch.einsum('thv,thk->hvk',u[0,:128].float(),k[0,:128].float())
    torch.testing.assert_close(snapshot[0],expected,rtol=1e-5,atol=1e-6)
    assert torch.equal(snapshot[0].to(h.dtype),h[0,2])
    assert torch.equal(pool[1],original[1]) and torch.equal(pool[4],original[4])
    if dtype == torch.bfloat16:
        assert torch.any(snapshot[0] != snapshot[0].bfloat16().float())


def test_snapshot_requires_paired_fp32_buffers():
    k=torch.zeros(1,64,1,64,device='cuda',dtype=torch.bfloat16)
    with pytest.raises(AssertionError):
        chunk_gated_delta_rule_fwd_h(k,k,k,track_state=torch.zeros(1,1,64,64,device='cuda'))
    with pytest.raises(AssertionError):
        chunk_gated_delta_rule_fwd_h(k,k,k,track_state=k,track_chunk_idx=torch.zeros(1,device='cuda',dtype=torch.int32))


import torch
from sglang.srt.layers.attention.linear.kernels.kda_triton import TritonKDAKernel
from hcu_adapter.fla_hcu_router import _install_kda_prefill


def test_live_extend_writes_snapshot():
    _install_kda_prefill("fork")
    torch.manual_seed(17)

    dev='cuda'; H=2; K=128; T=193
    q,k,v=[torch.randn(1,T,H,K,device=dev,dtype=torch.bfloat16) for _ in range(3)]
    g=torch.randn_like(q); beta=torch.randn(1,T,H,device=dev,dtype=torch.bfloat16)
    states=torch.randn(4,H,K,K,device=dev,dtype=torch.bfloat16)*0.1
    track=torch.full((1,H,K,K),float('nan'),device=dev,dtype=torch.float32)
    kwargs=dict(ssm_states=states,cache_indices=torch.tensor([2],device=dev,dtype=torch.int32),query_start_loc=torch.tensor([0,T],device=dev,dtype=torch.int32),A_log=torch.zeros(H,device=dev),dt_bias=torch.zeros(H*K,device=dev),lower_bound=-5.,beta_is_raw=True,return_intermediate_states=True,track_state=track,track_chunk_idx=torch.tensor([2],device=dev,dtype=torch.int32))
    out,h=TritonKDAKernel().extend(q=q,k=k,v=v.clone(),g=g,beta=beta,**kwargs)
    assert torch.isfinite(track).all(), 'KDA cache snapshot was not written: NaN sentinel preserved'
    assert torch.equal(track[0].to(h.dtype),h[0,2]), 'snapshot is not the state entering tracked chunk'
    print('PASS: live TritonKDAKernel.extend wrote the tracked state')
