import torch
import torch.nn.functional as F

from pandemonium.demons import Demon, Loss, ControlDemon, PredictionDemon
from pandemonium.experience import Trajectory, Transitions
from pandemonium.utilities.utilities import get_all_classes


class OfflineTD(Demon):
    # TODO: consider renaming to MultistepTD to remove potential confusion
    #   with batch RL methods
    r""" Base class for forward-view :math:`\text{TD}` methods.

    This class is used as a base for most of the DRL algorithms due to
    synergy with batching.
    """

    def __init__(self, criterion=F.smooth_l1_loss, **kwargs):
        super().__init__(**kwargs)
        self.criterion = criterion

    def delta(self, trajectory: Trajectory) -> Loss:
        """ Updates a value of a state using information in the trajectory """
        raise NotImplementedError

    def target(self, *args, **kwargs):
        """ Computes discounted returns for each step in the trajectory """
        raise NotImplementedError

    def learn(self, transitions: Transitions):
        """

        As opposed to the online case, where we learn on individual
        `Transition`s, in the offline case we learn on a sequence of
        transitions referred to as `Trajectory`.
        """
        trajectory = Trajectory.from_transitions(transitions)
        return self.delta(trajectory)


class OfflineTDPrediction(OfflineTD, PredictionDemon):
    r""" Offline :math:`\text{TD}(\lambda)` for prediction tasks """

    @torch.no_grad()
    def v_target(self, trajectory: Trajectory):
        """ Computes value targets from states in the trajectory """
        raise NotImplementedError

    def delta(self, trajectory: Trajectory) -> Loss:
        x = self.feature(trajectory.s0)
        v = self.predict(x)
        u = self.target(trajectory).detach()
        loss = self.criterion(input=v, target=u, reduction='none')
        loss = (loss * trajectory.ρ).mean()  # weighted IS
        return loss, {'loss': loss.item(), 'td_error': u - v}

    def target(self, trajectory: Trajectory):
        return super().target(trajectory, v=self.v_target(trajectory))


class OfflineTDControl(OfflineTD, ControlDemon):

    @torch.no_grad()
    def v_target(self, trajectory: Trajectory):
        """ Computes value targets from action-value pairs in the trajectory """
        raise NotImplementedError

    def eval_actions(self, x, a):
        """ Computes values associated with actions `a`

        Is overridden by distributional agents that use `azf`

        Returns
        -------
        (N, 1) vector of values for each action in the `a` vector.
        """
        return self.aqf(x)[torch.arange(a.size(0)), a].unsqueeze(1)

    def delta(self, trajectory: Trajectory) -> Loss:
        x = self.feature(trajectory.s0)  # could use trajectory.x1 instead
        v = self.eval_actions(x, trajectory.a)
        u = self.target(trajectory).detach()
        assert u.shape == v.shape, f'{u.shape} vs {v.shape}'
        loss = self.criterion(input=v, target=u, reduction='none')
        if len(loss.shape) > 2:
            # Pixel Control task?
            # take average across states
            loss = loss.mean(tuple(range(2, len(loss.shape))))
        batch_loss = (loss * trajectory.ρ).mean()  # weighted IS
        # TODO: td-error is not necessarily u-v depending on the criterion
        return batch_loss, {'batch_loss': batch_loss.item(),
                            'loss': loss,
                            'td_error': u - v}

    def target(self, trajectory: Trajectory):
        return super().target(trajectory, v=self.v_target(trajectory))


class TTD(OfflineTD):
    r""" Truncated :math:`\text{TD}(\lambda)`

    Notes
    -----
    Generalizes $n$-step $\text{TD}$ by allowing arbitrary mixing of
    $n$-step returns via $\lambda$ parameter.

    Depending on the algorithm, vector `v` would contain different
    bootstrapped estimates of values:

    .. math::
        - \text{TD}(\lambda) (forward view): state value estimates \text{V_t}(s)
        - \text{Q}(\lambda): action value estimates \max\limits_{a}(Q_t(s_t, a))
        - \text{SARSA}(\lambda): action value estimates Q_t(s_t, a_t)
        - \text{CategoricalQ}: atom values of the distribution

    The resulting vector `u` contains target returns for each state along
    the trajectory, with $V(S_i)$ for $i \in \{0, 1, \dots, n-1\}$ getting
    updated towards $[n, n-1, \dots, 1]$-step $\lambda$ returns respectively.

    References
    ----------
    Sutton and Barto (2018) ch. 12.3, 12.8, equation (12.18)
        http://incompleteideas.net/book/the-book.html
    van Seijen (2016) Appendix B, https://arxiv.org/pdf/1608.05151v1.pdf
        https://github.com/deepmind/rlax/blob/master/rlax/_src/multistep.py#L33
    """

    def target(self, trajectory: Trajectory, v: torch.Tensor):
        assert len(trajectory) == v.shape[0]
        γ = self.gvf.continuation(trajectory)
        z = self.gvf.cumulant(trajectory)
        λ = self.λ(trajectory)
        g = v[-1]
        u = torch.empty_like(v, dtype=torch.float)
        for i in range(len(trajectory) - 1, -1, -1):
            g = u[i] = z[i] + γ[i] * ((1 - λ[i]) * v[i] + λ[i] * g)
        return u


class TDn(TTD):
    r""" :math:`\text{n-step TD}` for estimating :math:`V ≈ v_{\pi}`

    Targets are calculated using forward view from $n$-step returns, where
    $n$ is determined by the length of trajectory. $\text{TDn}$ is a special
    case of truncated $\text{TD}$ with $\lambda=1$.

    The actual value of `n` is determined implicitly from the length of the
    trajectory (which itself is based on the `rollout_fragment_length`).

    TODO: clarify the relationship between n-step, rollout_fragment_length,
        batch_size, training_iteration

    """

    def __init__(self, **kwargs):
        super().__init__(
            eligibility=lambda trajectory: torch.ones_like(trajectory.r),
            **kwargs
        )


__all__ = get_all_classes(__name__)
