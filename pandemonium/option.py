from pandemonium.continuations import ContinuationFunction
from pandemonium.policies import Policy


class Option:
    def __init__(self,
                 initiation: callable,
                 continuation: ContinuationFunction,
                 policy: Policy
                 ):
        self.I = self.initiation = initiation
        self.β = self.continuation = continuation
        self.π = self.policy = policy

    def __repr__(self):
        return f'Option(\n' \
               f'\t(Ι): {self.I}\n' \
               f'\t(β): {self.β}\n' \
               f'\t(π): {self.π}\n' \
               f')'
