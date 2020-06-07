import torch


class EligibilityTrace:
    r""" Base class for various eligibility traces in :math:`\TD` learners.

    Eligibility trace is a mechanism for a short-term memory,
    mathematically represented as a vector, $e_t \in \mathbb{R}^d$,
    that parallels the long-term weight vector $w_t \in \mathbb{R}^d$.

    The rough intuition is that when a component of $w_t$ participates in
    producing an estimated value, then the corresponding component of $e_t$
    is bumped up and then begins to fade away. Learning will then occur in
    that component of $w_t$ if a nonzero $\TD$ error occurs before the trace
    falls back to zero.

    The trace-decay parameter $\lambda \mapsto [0, 1]$ determines the rate
    at which the trace falls.
    """

    def __init__(self, λ, trace_dim):
        assert 0 <= λ <= 1
        self.λ = self.trace_decay = λ
        self.trace = self.eligibility = self.e = torch.zeros(trace_dim)

    def __call__(self, *args, **kwargs):
        raise NotImplementedError

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return f'{self.__class__.__name__}({self.λ})'


class AccumulatingTrace(EligibilityTrace):

    def __call__(self, gamma, grad):
        self.e = self.λ * gamma * self.e + grad
        return self.e


class DutchTrace(EligibilityTrace):

    def __call__(self, gamma, x, alpha, *args, **kwargs):
        λ, γ, α = self.λ, gamma, alpha
        self.e = λ * γ * self.e + (1 - α * λ * γ * self.e.dot(x)) * x
        return self.e


class Vtrace(EligibilityTrace):
    pass


class Retrace(EligibilityTrace):
    pass
