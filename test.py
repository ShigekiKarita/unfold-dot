import torch
from torch.autograd.gradcheck import gradcheck

import unfold_dot
import unfold_dot_cuda


def reference_unfold_matmul(a, v):
    """
    Arguments:
      a: (batch, head, time, restrict)
      v: (batch, head, time, feat) -> will be windowed (unfolded) to (batch, head, time, restrict, feat)
    Returns:
      (batch, head, time, feat)
    """
    restrict = a.shape[-1]
    batch, head, time, feat = v.shape
    assert a.shape == (batch, head, time, restrict)
    unfold = torch.nn.Unfold(kernel_size=(restrict, 1), stride=(1, 1), padding=(restrict // 2, 0))
    v = unfold(v.transpose(2, 3).contiguous().view(batch, head * feat, -1, 1))
    v = v.view(batch, head, feat, restrict, time).transpose(2, 4)  # (batch, head, time, restrict, feat)
    # (b, h, t, 1, r) x (b, h, t, r, f) -> (b, h, t, 1, f)
    return a.unsqueeze(3).matmul(v).squeeze(3)


for w in [1, 3, 5]:
    q = torch.randn(2, 5, 7, 6, requires_grad=True, device="cuda")
    k = torch.randn(2, 5, 7, 6, requires_grad=True, device="cuda")

    s_ref = unfold_dot.reference_unfold_dot(q, k, w)
    s = unfold_dot_cuda.unfold_dot_cuda_forward(q, k, w)
    torch.testing.assert_allclose(s_ref, s)

    # test backward
    ds = torch.randn(*s.shape, device="cuda")
    s_ref.backward(ds)
    dq, dk = unfold_dot_cuda.unfold_dot_cuda_backward(ds, q, k)
    torch.testing.assert_allclose(q.grad, dq)
    torch.testing.assert_allclose(k.grad, dk)

    # test strided/non-contiguous tensors
    q = torch.randn(2, 6, 5, 7, device="cuda")
    k = torch.randn(2, 5, 6, 7, device="cuda")
    q = q.permute(0, 2, 3, 1)
    assert not q.is_contiguous()
    k = k.permute(0, 1, 3, 2)
    assert not k.is_contiguous()
    q.requires_grad = True
    k.requires_grad = True
    s_ref = unfold_dot.reference_unfold_dot(q, k, 3)
    s = unfold_dot_cuda.unfold_dot_cuda_forward(q, k, 3)
    torch.testing.assert_allclose(s_ref, s)
    ds = torch.randn(*s.shape, device="cuda")
    ds = ds.transpose(1, 2).contiguous()
    ds = ds.transpose(1, 2)
    assert not ds.is_contiguous()
    s_ref.backward(ds)
    dq, dk = unfold_dot_cuda.unfold_dot_cuda_backward(ds, q, k)
    torch.testing.assert_allclose(q.grad, dq)
    torch.testing.assert_allclose(k.grad, dk)


q = torch.randn(2, 3, 7, 5, requires_grad=True, device="cuda", dtype=torch.double)
k = torch.randn(2, 3, 7, 5, requires_grad=True, device="cuda", dtype=torch.double)
func = unfold_dot.UnfoldDot(3, True)
assert gradcheck(func, [q, k], eps=1e-3) # , atol=1e-2, rtol=1e-2)


# test matmul
restrict = 3
a = torch.randn(2, 3, 7, restrict, requires_grad=True, device="cuda", dtype=torch.double)
v = torch.randn(2, 3, 7, 5, requires_grad=True, device="cuda", dtype=torch.double)
av_ref = reference_unfold_matmul(a, v)
av = unfold_dot_cuda.unfold_matmul_cuda_forward(a, v)
torch.testing.assert_allclose(av_ref, av)

dav = torch.randn(*av.shape, device="cuda", dtype=torch.double)
av_ref.backward(dav)
da, dv = unfold_dot_cuda.unfold_matmul_cuda_backward(dav, a, v)
torch.testing.assert_allclose(a.grad, da)
