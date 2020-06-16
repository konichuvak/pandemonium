from collections import OrderedDict
from functools import partial
from warnings import warn

import torch
from torch import nn

from pandemonium.demons import Loss, ParametricDemon, ControlDemon
from pandemonium.demons.offline_td import OfflineTDControl
from pandemonium.demons.offline_td import TTD, TDn, OfflineTDPrediction
from pandemonium.experience import (ER, PER, ReplayBufferMixin, Trajectory,
                                    Transitions)
from pandemonium.networks import Reshape, TargetNetMixin
from pandemonium.policies import Policy, HierarchicalPolicy, DiffPolicy
from pandemonium.policies.utils import torch_argmax_mask
from pandemonium.utilities.distributions import cross_entropy, l2_projection
from pandemonium.utilities.utilities import get_all_classes


class DuellingMixin:
    r""" Mixin for value-based control algorithms that uses two separate
    estimators for action-values and state-values.

    References
    ----------
    "Dueling Network Architectures for DRL" by Wang et al.
        https://arxiv.org/pdf/1511.06581

    """
    avf: callable
    aqf: callable
    target_avf: callable
    target_aqf: callable

    def __init__(self):
        if not isinstance(self, ControlDemon):
            raise TypeError(f'Duelling architecture is only supported'
                            f'by control algorithms')
        self.avf = nn.Linear(self.φ.feature_dim, 1)

    def predict_q(self, x):
        v, q = self.avf(x), self.aqf(x)
        return q - (q - v).mean(1, keepdim=True)

    def predict_target_q(self, x):
        q, v = self.target_avf(x), self.target_aqf(x)
        return q - (q - v).mean(1, keepdim=True)


class CategoricalQ:
    """ Categorical Q-learning mixin.

    References
    ----------
    "A Distributional Perspective on RL" by Bellemare et al.
        https://arxiv.org/abs/1707.06887
    """
    φ: callable
    μ: Policy

    def __init__(self,
                 num_atoms: int,
                 v_min: float,
                 v_max: float):
        """ Sets up functions that allow for distributional value learning.

        Parameters
        ----------
        num_atoms: int
            Number of atoms (bins) for representing the distribution of return.
            When this is greater than 1, distributional Q-learning is used.
        v_min: float
            Minimum possible Q-value of the distribution
        v_max: float
            Maximum possible Q-value of the distribution
        """
        assert num_atoms > 0, f'num_atoms should be greater than 0, got {num_atoms}'
        assert v_max > v_min, f'{v_max} should be greater than {v_min}'
        self.num_atoms = num_atoms
        self.v_min, self.v_max = v_min, v_max
        self.z = self.atoms = torch.linspace(v_min, v_max, num_atoms)
        self.support = self.atoms[None, :]
        self.Δz = (v_max - v_min) / float(num_atoms - 1)
        self.azf = self.target_azf = nn.Sequential(
            nn.Linear(self.φ.feature_dim,
                      self.μ.action_space.n * num_atoms),
            Reshape(-1, self.μ.action_space.n, num_atoms),
            nn.Softmax(dim=2)
        )
        self.criterion = cross_entropy

        del self.aqf  # necessary when aqf is a torch net

        def implied_aqf(x):
            r""" Computes expected action-values using the learned distribution.

            Overrides ``pandemonium.demons.demon.ControlDemon.aqf`` with an
            aqf induced by the distributional action-value function `azf`.

            .. math::
                Q(x, a) = \sum_{i=1}^{N} z_i p_i(x, a)
            """
            return torch.einsum('k,ijk->ij', [self.atoms, self.azf(x)])

        self.aqf = implied_aqf

        def eval_actions(x, a):
            r""" Computes the value distribution for given actions.

            Overrides
            ``pandemonium.demons.offline_td.OfflineTDControl.eval_actions``.

            Returns
            -------
            (N, n_atoms) matrix with probability vectors as rows, inducing
            a probability mass function for values of actions in `a`.
            """
            return self.azf(x)[torch.arange(a.size(0)), a]

        self.eval_actions = eval_actions

        __delta = self.delta

        def delta(trajectory: Trajectory) -> Loss:
            batch_loss, info = __delta(trajectory)
            info['ce_loss'] = info.pop('loss')  # rename for clarity
            del info['td_error']
            return batch_loss, info

        self.delta = delta

        @torch.no_grad()
        def target(trajectory: Trajectory):
            r""" Computes value targets using Categorical Q-learning.

            Effectively reduces the usual Bellman update to multi-class
            classification problem.

            On the implementation side, operating on distributions would be
            nice too. However torch distributions API is quite limited atm.
                Z = Categorical(probs=self.target_azf(trajectory.x1))
                T = [AffineTransform(loc=r, scale=γ, event_dim=1)]
                TZ = TransformedDistribution(base_distribution=Z, transforms=T)
            """
            assert isinstance(self, (DQN, TDn))  # make linter happy

            # Scale and shift the support with the multi-step Bellman operator
            # TZ(x,a) = R(x, a) + γZ(x',a')
            support = self.support.repeat_interleave(len(trajectory), dim=0)
            Tz = TDn.target(self, trajectory, v=support)  # (batch, num_atoms)

            # Compute probability mass vector for greedy action in the next state
            Z = self.target_azf(trajectory.x1)  # (batch, actions, atoms)
            q = torch.einsum('k,ijk->ij', [self.atoms, Z])  # (batch, actions)
            if self.double:
                online_q = self.aqf(trajectory.x1)
                a = torch_argmax_mask(online_q, 1).long().argmax(1)
            else:
                a = q.argmax(1)
            probs = Z[torch.arange(len(trajectory)), a]
            assert torch.allclose(probs.sum(1), torch.ones(probs.size(0)))

            # Compute ΦΤz
            # Projects the target distribution (Tz, probs) with a shifted support
            # `Tz` and probability mass vector `probs` onto the support of our
            # parametric model using the l2 (a.k.a. Cramer) distance
            projected_probs = l2_projection(Tz, probs, self.atoms)
            return projected_probs

        self.target = target


class DQN(OfflineTDControl,
          TDn,
          ParametricDemon,
          ReplayBufferMixin,
          TargetNetMixin,
          CategoricalQ,
          DuellingMixin):
    """ Deep Q-Network with all the bells and whistles mixed in.

    References
    ----------
    "Rainbow: Combining Improvements in Deep RL" by Hessel et. al
        https://arxiv.org/pdf/1710.02298.pdf
    """

    def __init__(self,
                 feature: callable,
                 behavior_policy: Policy,
                 replay_buffer: ER,
                 target_update_freq: int = 0,
                 warm_up_period: int = None,
                 num_atoms: int = 1,
                 v_min: float = None,
                 v_max: float = None,
                 duelling: bool = False,
                 double: bool = False,
                 **kwargs):

        # Adds a replay buffer
        priority = None
        if isinstance(replay_buffer, PER):
            # Use cross-entropy loss as a measure of priority
            priority = 'ce_loss' if num_atoms > 1 else 'td_error'
        ReplayBufferMixin.__init__(self, replay_buffer, priority)

        # By default, learning does not start until the replay buffer is full
        if warm_up_period is None:
            warm_up_period = replay_buffer.capacity // replay_buffer.batch_size
        self.warm_up_period = warm_up_period

        # Initialize Q-network demon
        aqf = nn.Linear(feature.feature_dim, behavior_policy.action_space.n)
        super(DQN, self).__init__(aqf=aqf, feature=feature,
                                  behavior_policy=behavior_policy, **kwargs)

        # Replaces `avf` implied by `aqf` with an independent estimator
        self.duelling = duelling
        if duelling:
            DuellingMixin.__init__(self)

        # Adds ability to approximate expected values
        # via learning a distribution
        if num_atoms > 1:
            CategoricalQ.__init__(self, num_atoms=num_atoms,
                                  v_min=v_min, v_max=v_max)

        # Value-based policies require a value function to determine the
        # preference for actions. Since we now know what our action-value
        # function is (after possibly being affected by CategoricalQ mixin),
        # we can pass it to the policy via `partial`
        self.μ.act = partial(self.μ.act, q_fn=self.aqf)

        # Adds a target network to stabilize SGD
        TargetNetMixin.__init__(self, target_update_freq)
        if self.target_aqf == self.aqf:
            # TODO: fix the distributional case
            warn('target aqf == aqf')
        if self.target_avf == self.avf:
            warn('target avf == avf')

        # Adds double Q-learning for tackling maximization bias
        self.double = double
        if double:
            assert target_update_freq > 0

    def predict_q(self, x, target: bool = False):
        return self.aqf(x) if not target else self.target_aqf(x)

    @torch.no_grad()
    def v_target(self, trajectory: Trajectory):
        q = self.predict_q(trajectory.x1, target=True)
        if self.double:
            if isinstance(self, PixelControl):
                # TODO: q[mask] returns flat obs
                raise NotImplementedError
            v = q[torch_argmax_mask(self.aqf(trajectory.x1), 1)].unsqueeze(-1)
        else:
            v = q.max(1, keepdim=True)[0]
        return v

    def learn(self, transitions: Transitions):

        self.store(transitions)
        self.sync_target()

        # Wait until warm up period is over
        self.update_counter += 1  # TODO: move the counter up the hierarchy?
        if self.update_counter < self.warm_up_period:
            return None, dict()

        # Learn from experience
        # TODO: differentiate between n-step and batched one-step
        #   We can EITHER sample n transitions at random from a replay buffer
        #   and do a batched one-step backup on them OR we can sample n
        #   consequent transition and do a multistep update with them.
        #   This should be controlled via `n_step` parameter passed to DQN upon
        #   initialization.
        transitions = self.replay_buffer.sample()
        if not transitions:
            return None, dict()  # not enough experience in the buffer
        trajectory = Trajectory.from_transitions(transitions)
        δ, info = self.delta(trajectory)

        # Update the priorities of the collected transitions
        if isinstance(self.replay_buffer, PER):
            self._update_priorities(trajectory, info)

        return δ, info

    def __repr__(self):
        demon = ParametricDemon().__repr__()
        params = f'(replay_buffer): {repr(self.replay_buffer)}\n' \
                 f'(warmup): {self.warm_up_period}\n' \
                 f'(target_update_freq): {self.target_update_freq}\n' \
                 f'(double): {self.double}\n' \
                 f'(duelling): {self.duelling}\n'
        return f'{demon}\n{params}'

    def __str__(self):
        return super().__str__()


class DeepSARSA(OfflineTDControl):
    r""" :math:`n`-step semi-gradient :math:`\SARSA` """

    @torch.no_grad()
    def v_target(self, trajectory: Trajectory):
        q = self.predict_q(trajectory.x1)
        a = self.gvf.π(trajectory.x1, vf=self.aqf)
        v = q[torch.arange(q.size(0)), a]
        return v


class DeepSARSE(OfflineTDControl):
    r""" :math:`n`-step semi-gradient expected :math:`\SARSA` """

    @torch.no_grad()
    def v_target(self, trajectory: Trajectory):
        q = self.predict_q(trajectory.x1)
        dist = self.gvf.π.dist(trajectory.x1, vf=self.aqf)
        v = q * dist.probs
        return v



__all__ = get_all_classes(__name__)
