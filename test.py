import torch
from torch.autograd.gradcheck import gradcheck

import unfold_dot
import unfold_dot_cuda

# test dot
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
    func = unfold_dot.UnfoldDot(w, True)
    assert gradcheck(func, [q, k], eps=1e-3) # , atol=1e-2, rtol=1e-2)


# test matmul
for restrict in [1, 3, 5]:
    a = torch.randn(2, 3, 7, restrict, requires_grad=True, device="cuda", dtype=torch.double)
    v = torch.randn(2, 3, 7, 5, requires_grad=True, device="cuda", dtype=torch.double)
    av_ref = unfold_dot.reference_unfold_matmul(a, v)
    av = unfold_dot_cuda.unfold_matmul_cuda_forward(a, v)
    torch.testing.assert_allclose(av_ref, av)

    dav = torch.randn(*av.shape, device="cuda", dtype=torch.double)
    av_ref.backward(dav)
    da, dv = unfold_dot_cuda.unfold_matmul_cuda_backward(dav, a, v)
    torch.testing.assert_allclose(a.grad, da)
    torch.testing.assert_allclose(v.grad, dv)

    # test strided/non-contiguous tensors
    a = torch.randn(2, 3, 7, restrict, device="cuda")
    v = torch.randn(2, 3, 7, 10, device="cuda")
    a = a.transpose(1, 2).contiguous()
    a = a.transpose(1, 2)
    assert not a.is_contiguous()
    v = v.transpose(3, 0).contiguous()
    v = v.transpose(3, 0)
    assert not v.is_contiguous()
    a.requires_grad = True
    v.requires_grad = True
    av_ref = unfold_dot.reference_unfold_matmul(a, v)
    av = unfold_dot_cuda.unfold_matmul_cuda_forward(a, v)
    torch.testing.assert_allclose(av_ref, av)

    dav = torch.randn(*av.shape, device="cuda")
    dav = dav.transpose(0, 2).contiguous()
    dav = dav.transpose(0, 2)
    assert not dav.is_contiguous()
    av_ref.backward(dav)
    da, dv = unfold_dot_cuda.unfold_matmul_cuda_backward(dav, a, v)
    torch.testing.assert_allclose(a.grad, da)
    torch.testing.assert_allclose(v.grad, dv)

    a = torch.randn(2, 3, 7, restrict, requires_grad=True, device="cuda", dtype=torch.double)
    v = torch.randn(2, 3, 7, 5, requires_grad=True, device="cuda", dtype=torch.double)
    func = unfold_dot.UnfoldMatmul()
    assert gradcheck(func, [a, v], eps=1e-3)




mha = unfold_dot.MultiHeadedAttention(2, 6, 0.0, 3)
mha.cuda()
q = torch.randn(2, 5, 6, requires_grad=True, device="cuda")
k = torch.randn(2, 5, 6, requires_grad=True, device="cuda")
v = torch.randn(2, 5, 6, requires_grad=True, device="cuda")
m = torch.ones(2, 5, dtype=torch.uint8, device="cuda")
y_ref = mha.reference_forward(q, k, v, m)
y = mha(q, k, v, m)
torch.testing.assert_allclose(y_ref, y)
