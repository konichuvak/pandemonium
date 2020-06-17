from functools import partial

import torch
import torch.nn.functional as F
from torch import nn

from pandemonium.demons import Loss, ParametricDemon, ControlDemon
from pandemonium.demons.offline_td import OfflineTD, TDn
from pandemonium.demons.online_td import OnlineTD
from pandemonium.experience import Trajectory, Experience, Transition
from pandemonium.networks import Reshape
from pandemonium.policies import Policy, Egreedy
from pandemonium.policies.utils import torch_argmax_mask
from pandemonium.utilities.distributions import cross_entropy, l2_projection
from pandemonium.utilities.utilities import get_all_classes


class TDControl(ParametricDemon, ControlDemon):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Policies require a value function to determine action preferences
        self.μ.act = partial(self.μ.act, q_fn=self.predict_q)

    def q_tm1(self, x, a):
        """ Computes values associated with action batch `a`

        Is overridden by distributional agents that use `azf` to evaluate
        actions instead of `aqf`.

        Returns
        -------
        A batch of action values $Q(s_{t-1}, a_{t-1})$.
        """
        return self.aqf(x)[torch.arange(a.size(0)), a]

    @torch.no_grad()
    def q_t(self, exp: Experience):
        r""" Computes action-value targets :math:`Q(s_{t+1}, \hat{a})`.

        Algorithms differ in the way $\hat{a}$ is chosen.

        .. math::
            \begin{align*}
                \text{Q-learning} &: \hat{a} = \argmax_{a \in \mathcal{A}}Q(s_{t+1}, a) \\
                \SARSA &: \hat{a} = \mu(s_{t+1})
            \end{align*}
        """
        raise NotImplementedError

    def behavior_policy(self, x: torch.Tensor):
        return self.μ(x, q_fn=self.aqf)

    def target(self, exp: Experience):
        return super().target(exp, v=self.q_t(exp))


class OnlineTDControl(TDControl, OnlineTD):
    r""" Base class for online :math:`\TD` methods for control tasks. """

    def delta(self, t: Transition) -> Loss:
        γ = self.gvf.continuation(t)
        z = self.gvf.z(t)
        v = self.q_tm1(t.x0, t.a)
        u = z + γ * self.q_t(t)
        δ = u - v

        info = {'td_error': δ.item()}
        if self.λ.trace_decay == 0:
            loss = F.mse_loss(input=v, target=u)  # a shortcut
        else:
            v.backward()  # semi-gradient
            assert self.aqf.bias is None
            grad = next(self.aqf.parameters()).grad
            e = self.λ(γ, grad)
            info['eligibility_norm'] = e.pow(2).sum().sqrt().item()
            with torch.no_grad():
                for param in self.aqf.parameters():
                    param.grad = -δ * e
            loss = None

        return loss, info


class OfflineTDControl(TDControl, OfflineTD):
    r""" Offline :math:`\TD` for control tasks. """

    def delta(self, trajectory: Trajectory) -> Loss:
        x = self.feature(trajectory.s0)  # could use trajectory.x1 instead
        v = self.q_tm1(x, trajectory.a)
        u = self.target(trajectory).detach()
        assert u.shape == v.shape, f'{u.shape} vs {v.shape}'
        loss = self.criterion(input=v, target=u, reduction='none')
        loss = loss.view(len(trajectory), -1)
        batch_loss = (loss * trajectory.ρ).mean()  # weighted IS
        # TODO: td-error is not necessarily u-v depending on the criterion
        return batch_loss, {'batch_loss': batch_loss.item(),
                            'loss': loss,
                            'td_error': u - v}


class SARSA(TDControl):
    r""" Semi-gradient :math:`\SARSA{(\lambda)}`.

    References
    ----------
    "Reinforcement Learning: An Introduction"
        Sutton and Barto (2018) ch. 12.7
        http://incompleteideas.net/book/the-book.html

    """

    @torch.no_grad()
    def q_t(self, exp: Experience):
        q = self.predict_q(exp.x1)
        return q[torch.arange(q.size(0)), exp.a1]


class SARSE(TDControl):
    r""" Semi-gradient Expected :math:`\SARSA{(\lambda)}`.

    References
    ----------
    "Reinforcement Learning: An Introduction"
        Sutton and Barto (2018) ch. 6.6
        http://incompleteideas.net/book/the-book.html

    "A Theoretical and Empirical Analysis of Expected Sarsa"
        Harm van Seijen et al (2009)
        http://www.cs.ox.ac.uk/people/shimon.whiteson/pubs/vanseijenadprl09.pdf
    """

    @torch.no_grad()
    def q_t(self, exp: Experience):
        q = self.predict_q(exp.x1)
        dist = self.gvf.π.dist(exp.x1, q_fn=self.aqf)
        return torch.einsum('ba,ba->b', q, dist.probs)


class QLearning(TDControl):
    r""" Classic Q-learning update rule.

    Notes
    -----
    Can be interpreted as an off-policy version of :math:`\SARSE`.
    Since the target policy $\pi$ in canonical Q-learning is greedy wrt to GVF,
    we have the following equality:

    .. math::
        \max_\limits{a \in \mathcal{A}}Q(S_{t+1}, a) = \sum_{a \in \mathcal{A}} \pi(a|S_{t+1})Q(S_{t+1}, a)

    In this case the target Q-value estimator would be:

    .. code-block:: python

        @torch.no_grad()
        def q_t(self, exp: Experience):
            q = self.target_aqf(exp.x1)
            dist = self.gvf.π.dist(exp.x1, q_fn=self.aqf)
            return torch.einsum('ba,ba->b', q, dist.probs)

    We do not actually use this update in here since taking a max is more
    efficient than computing weights and taking a dot product.

    TODO: integrate
        online:
            duelling
        offline:
            duelling
            traces
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Ensures that target policy is greedy wrt to the Q function
        if not isinstance(self.gvf.π, Egreedy):
            raise TypeError(self.gvf.π)
        elif self.gvf.π.ε == 0:
            raise ValueError(self.gvf.π.ε)

    @torch.no_grad()
    def q_t(self, exp: Experience):
        q = self.predict_q(exp.x1)
        return q.max(1)[0]


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
        if not isinstance(self.avf, torch.nn.Module):
            self.avf = nn.Linear(self.φ.feature_dim, 1)

        def predict_q_(x):
            v, q = self.avf(x), self.aqf(x)
            return q - (q - v).mean(1, keepdim=True)

        self.predict_q = predict_q_

        def predict_target_q_(x):
            v, q = self.target_avf(x), self.target_aqf(x)
            return q - (q - v).mean(1, keepdim=True)

        self.predict_target_q = predict_target_q_


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


__all__ = get_all_classes(__name__)
