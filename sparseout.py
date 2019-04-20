from torch.autograd.function import InplaceFunction
from torch.autograd import Variable

r"""Implementation of Sparseout

    Examples::
    >>> import torch
    >>> import torch.nn.functional as F
    >>> from sparseout import sparseout
    >>> from torch.autograd import Variable
    
    >>> input = Variable(torch.randn(5, 5))
    >>> # Dropout
    >>> output_dropout = F.dropout(input, p=0.5)
    >>> # Sparsout
    >>> output_sparseout = sparseout(input, p=0.5, q=2.0)
    
.. _Sparseout: Controlling Sparsity in Deep Networks: https://arxiv.org/abs/1904.08050
"""


def sparseout(input, p, q, training=True, inplace=False):
    return SparseoutLayer.apply(input, p, q, training, inplace)


class SparseoutLayer(InplaceFunction):

    @staticmethod
    def _make_noise(input):
        return input.new().resize_as_(input)

    @classmethod
    def forward(cls, ctx, input, p=0.5, q=2.0, train=False, inplace=False):
        EPSILON=1E-12
        if p < 0 or p > 1:
            raise ValueError("dropout probability has to be between 0 and 1, "
                             "but got {}".format(p))
        if q < 0:
            raise ValueError("norm q has to be non-negative, "
                             "but got {}".format(q))
        ctx.p = p
        ctx.q = (q-2.0)/2.0
        ctx.train = train
        ctx.inplace = inplace

        if ctx.inplace:
            ctx.mark_dirty(input)
            output = input
        else:
            output = input.clone()

        if ctx.p > 0 and ctx.train:
            ctx.noise = cls._make_noise(input)
            if ctx.p == 1:
                ctx.noise.fill_(0)
            else:
                ctx.noise.bernoulli_(1 - ctx.p).div_(1 - ctx.p).sub_(1)
                ctx.noise = input.abs().add(EPSILON).pow(ctx.q).mul(ctx.noise).add(1)
                
            ctx.noise = ctx.noise.expand_as(input)
            output.mul_(ctx.noise)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.p > 0 and ctx.train:
            return grad_output.mul(Variable(ctx.noise)), None, None, None, None
        else:
            return grad_output, None, None, None, None
