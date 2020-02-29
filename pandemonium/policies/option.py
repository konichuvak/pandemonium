from pandemonium.continuations import ContinuationFunction
from pandemonium.policies import Policy


class Option:
    r""" A policy (decision making rule) that can activate and terminate

    .. todo::
        Consider making this a base class instead of the `Policy`
        policy is just a special case of option that can be initiated anywhere
        and is terminating nowhere.
    """

    def __init__(self,
                 initiation: callable,
                 continuation: ContinuationFunction,
                 policy: Policy
                 ):
        self.ι = self.initiation = initiation
        self.β = self.continuation = continuation
        self.π = self.policy = policy

    def __repr__(self):
        return f'Option(\n' \
               f'\t(ι): {self.ι}\n' \
               f'\t(β): {self.β}\n' \
               f'\t(π): {self.π}\n' \
               f')'
