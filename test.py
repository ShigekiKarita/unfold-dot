import torch
from torch.autograd.gradcheck import gradcheck

import unfold_dot
import unfold_dot_cuda


for w in [1, 3, 5]:
    q = torch.randn(2, 3, 7, 6, requires_grad=True, device="cuda")
    k = torch.randn(2, 3, 7, 6, requires_grad=True, device="cuda")

    s_ref = unfold_dot.reference_unfold_dot(q, k, w)
    s = unfold_dot_cuda.unfold_dot_cuda_forward(q, k, w)
    torch.testing.assert_allclose(s_ref, s.unsqueeze(3))

    # test backward
    s_ref.backward(torch.ones_like(s_ref))
    dq, dk = unfold_dot_cuda.unfold_dot_cuda_backward(torch.ones_like(s), q, k)
    torch.testing.assert_allclose(q.grad, dq)
    torch.testing.assert_allclose(k.grad, dk)

# FIXME: cuda gradcheck will fail (reference impl is good)
q = torch.randn(1, 1, 3, 2, requires_grad=True, dtype=torch.double).cuda()
k = torch.randn(1, 1, 3, 2, requires_grad=False, dtype=torch.double).cuda()
func = unfold_dot.UnfoldDot(3, True)
assert gradcheck(func, [q, k], eps=1e-3) # , atol=1e-2, rtol=1e-2)


q = torch.randn(1, 1, 3, 2, requires_grad=False, dtype=torch.double).cuda()
k = torch.randn(1, 1, 3, 2, requires_grad=True, dtype=torch.double).cuda()
func = unfold_dot.UnfoldDot(3, True)
assert gradcheck(func, [q, k], eps=1e-3) # , atol=1e-2, rtol=1e-2)
