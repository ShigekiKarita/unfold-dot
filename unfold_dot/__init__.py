import math
from torch import nn
from torch.autograd import Function
import torch


def reference_unfold_dot(q, k, restrict):
    # TODO use stride or padding to make time2 equal to time1
    n_batch = q.size(0)
    h = q.size(1)
    d_k = q.size(3)
    unfold = torch.nn.Unfold(kernel_size=(restrict, 1), stride=(1, 1), padding=(restrict // 2, 0))
    # (batch, self.h * self.d_k * self.restrict, time2)
    k = unfold(k.transpose(2, 3).contiguous().view(n_batch, h * d_k, -1, 1))
    # (batch, self.h, time2, self.d_k, self.restrict)
    k = k.view(n_batch, h, d_k, restrict, -1).permute(0, 1, 4, 2, 3)
    # (batch, head, time1, 1, d_k) x (batch, head, time1, d_k, self.restrict) -> (batch, head, time1, 1, self.restrict)
    scores = q.unsqueeze(-2).matmul(k) # / math.sqrt(self.d_k)
    return scores.squeeze(3)


class UnfoldDotFunction(Function):
    def __init__(self, restrict):
        super().__init__()
        self.restrict = restrict

    def forward(self, q, k):
        import unfold_dot_cuda
        ret = unfold_dot_cuda.unfold_dot_cuda_forward(q, k, self.restrict)
        self.save_for_backward(q, k)
        return ret

    def backward(self, dret):
        import unfold_dot_cuda
        q, k = self.saved_variables
        dq, dk = unfold_dot_cuda.unfold_dot_cuda_backward(dret, q, k)
        return dq, dk


class UnfoldDot(nn.Module):
    def __init__(self, restrict, faster=True):
        super().__init__()
        self.restrict = restrict
        self.faster = faster

    def forward(self, q, k):
        assert q.shape[2] == k.shape[2], "restricted attention is not implemented for source attention now"
        if self.faster and q.is_cuda and k.is_cuda:
            return UnfoldDotFunction(self.restrict)(q, k)
        else:
            return reference_unfold_dot(q, k, self.restrict)


def unfold_dot(q, k, restrict):
    """
    Args:
        q: (batch, head, time1, d_k)
        k: (batch, head, time1, d_k)
    Returns:
        (batch, head, time1, restrict)
    """
    return UnfoldDot(restrict)(q, k)


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


class UnfoldMatmulFunction(Function):
    @staticmethod
    def forward(ctx, a, v):
        import unfold_dot_cuda
        av = unfold_dot_cuda.unfold_matmul_cuda_forward(a, v)
        ctx.save_for_backward(a, v)
        return av

    @staticmethod
    def backward(ctx, dav):
        import unfold_dot_cuda
        a, v = ctx.saved_variables
        da, dv = unfold_dot_cuda.unfold_matmul_cuda_backward(dav, a, v)
        return da, dv


class UnfoldMatmul(nn.Module):
    def __init__(self, faster=True):
        super().__init__()
        self.faster = faster

    def forward(self, a, v):
        assert a.shape[:-1] == v.shape[:-1]
        if self.faster and a.is_cuda and v.is_cuda:
            return UnfoldMatmulFunction.apply(a, v)
        else:
            return reference_unfold_matmul(a, v)


def unfold_matmul(a, v, faster=True):
    """
    Arguments:
      a: (batch, head, time, restrict)
      v: (batch, head, time, feat) -> will be windowed (unfolded) to (batch, head, time, restrict, feat)
    Returns:
      (batch, head, time, feat)
    """
    return UnfoldMatmul(faster)(a, v)


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout, restrict=0):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linear_q = nn.Linear(d_model, d_model)
        self.linear_k = nn.Linear(d_model, d_model)
        self.linear_v = nn.Linear(d_model, d_model)
        self.linear_out = nn.Linear(d_model, d_model)
        self.att_ = None
        self.dropout = nn.Dropout(p=dropout)
        self.restrict = restrict if restrict is not None else 0
        if self.restrict > 0:
            assert self.restrict % 2 == 1

    @property
    def attn(self):
        if self.restrict > 0:
            # (batch, head, time1, 1, self.restrict)
            at = self.attn_.squeeze(3)
            batch, head, time, _ = at.shape
            a = self.attn_.new_zeros(batch, head, time + self.restrict, time)
            r = self.restrict // 2
            for t in range(time):
                a[:, :, t:t + self.restrict, t] = at[:, :, t]
            return a[:, :, r:-r-1, :]
        return self.attn_

    def forward(self, query, key, value, mask):
        """Compute 'Scaled Dot Product Attention'
        :param torch.Tensor query: (batch, time1, size)
        :param torch.Tensor key: (batch, time2, size)
        :param torch.Tensor value: (batch, time2, size)
        :param torch.Tensor mask: (batch, time1)
        :param torch.nn.Dropout dropout:
        :return torch.Tensor: attentined and transformed `value` (batch, time1, d_model)
             weighted by the query dot key attention (batch, head, time1, time2)
        """
        n_batch = query.size(0)
        q = self.linear_q(query).view(n_batch, -1, self.h, self.d_k)
        k = self.linear_k(key).view(n_batch, -1, self.h, self.d_k)
        v = self.linear_v(value).view(n_batch, -1, self.h, self.d_k)
        q = q.transpose(1, 2)  # (batch, head, time1, d_k)
        k = k.transpose(1, 2)  # (batch, head, time2, d_k)
        v = v.transpose(1, 2)  # (batch, head, time2, d_k)
        
        if self.restrict > 0:
            # TODO use stride or padding to make time2 equal to time1
            unfold = nn.Unfold(kernel_size=(self.restrict, 1), stride=(1, 1), padding=(self.restrict // 2, 0))
            # # (batch, self.h * self.d_k * self.restrict, time2)
            # k = unfold(k.transpose(2, 3).contiguous().view(n_batch, self.h * self.d_k, -1, 1))
            # # (batch, self.h, time2, self.d_k, self.restrict)
            # k = k.view(n_batch, self.h, self.d_k, self.restrict, -1).permute(0, 1, 4, 2, 3)
            # (batch, self.h * self.d_k * self.restrict, time2)
            v = unfold(v.transpose(2, 3).contiguous().view(n_batch, self.h * self.d_k, -1, 1))
            # (batch, self.h, time2, self.restrict, self.d_k)
            v = v.view(n_batch, self.h, self.d_k, self.restrict, -1).transpose(2, 4)
            # (batch, head, time1, 1, d_k) x (batch, head, time1, d_k, self.restrict) -> (batch, head, time1, 1, self.restrict)
            # scores = q.unsqueeze(-2).matmul(k) / math.sqrt(self.d_k)
            scores = unfold_dot(q, k, self.restrict).unsqueeze(3) / math.sqrt(self.d_k)
            if mask is not None:
                mask = mask.unsqueeze(-1).unsqueeze(-1)
                self.attn_ = torch.softmax(scores, dim = -1)  # (batch, head, time1, time2)
                self.attn_ = self.attn_.masked_fill(mask == 0, 0)
        else:
            # (batch, head, time1, d_k) x (batch, head, d_k, time2) -> (batch, head, time1, time2)
            scores = q.matmul(k.transpose(-2, -1)) / math.sqrt(self.d_k)
            if mask is not None:
                mask = mask.unsqueeze(1)
                scores = scores.masked_fill(mask == 0, MIN_VALUE)
            self.attn_ = torch.softmax(scores, dim = -1)  # (batch, head, time1, time2)
        p_attn = self.dropout(self.attn_)
        # TODO make this unfold_dot: (batch, head, time1, restrict) x (batch, self.h, time1, self.restrict, self.d_k)
        x = torch.matmul(p_attn, v)  # (batch, head, time1, d_k)
        x = x.transpose(1, 2).contiguous().view(n_batch, -1, self.h * self.d_k)  # (batch, time1, d_model)
        return self.linear_out(x) # (batch, time1, d_model)
