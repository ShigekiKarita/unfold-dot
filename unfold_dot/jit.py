from torch.utils.cpp_extension import load
unfold_dot_cuda = load(
    'unfold_dot_cuda', ['unfold_dot_cuda.cpp', 'unfold_dot_cuda_kernel.cu'], verbose=True)
help(unfold_dot_cuda)
