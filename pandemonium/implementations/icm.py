import torch.nn.functional as F
from torch import device

from pandemonium import GVF, Horde
from pandemonium.continuations import ConstantContinuation
from pandemonium.cumulants import CombinedCumulant, Fitness, Cumulant
from pandemonium.demons import Loss, ParametricDemon
from pandemonium.experience import Experience, Transition, Trajectory
from pandemonium.implementations import AC
from pandemonium.networks import ForwardModel, InverseModel
from pandemonium.policies import Policy, Greedy


class IntrinsicCuriosityModule(ParametricDemon):

    def __init__(self,
                 feature: callable,
                 behavior_policy: Policy,
                 beta: float,
                 ):

        super().__init__(
            gvf=None,  # not learning any value function
            avf=None,  # not approximating any value function
            feature=feature,
            behavior_policy=behavior_policy,
            eligibility=None,
        )
        self.inverse_dynamics_model = InverseModel(
            action_dim=behavior_policy.action_space.n,
            feature_dim=feature.feature_dim,
        )
        self.forward_dynamics_model = ForwardModel(
            action_dim=behavior_policy.action_space.n,
            feature_dim=feature.feature_dim
        )

        assert 0. <= beta <= 1.
        self.β = beta

    def learn(self, exp: Experience):
        if isinstance(exp, (Trajectory, Transition)):
            return self.delta(experience=exp)
        elif isinstance(exp, list) and isinstance(exp[0], Transition):
            trajectory = Trajectory.from_transitions(exp)
            return self.delta(experience=trajectory)

    def delta(self, experience: Experience) -> Loss:
        # Reuse forward model loss computed during cumulant signal generation
        φ0, φ1 = experience.info['φ0'], experience.info['φ1']
        forward_loss = experience.info['forward_model_error']

        # Predict the action from the current and the next feature vectors
        a_hat = self.inverse_dynamics_model(φ0, φ1)
        inverse_loss = F.cross_entropy(a_hat, experience.a)

        # Combine two losses via simple weighting
        info = {'id_loss': inverse_loss.item(), 'fd_loss': forward_loss.item()}
        return self.β * forward_loss + (1 - self.β) * inverse_loss, info


class Curiosity(Cumulant):
    r""" Tracks the novelty of the states wrt to the forward model

    Intrinsic reward on the transition at time $t$ is given by

    .. math::
        r^i_t = \frac{1}{2} \norm{\hat{\phi}(s_{t+1}) - \phi(s_{t+1})}^2_2

    References
    ----------
    "Curiosity-driven Exploration by Self-supervised Prediction"
        Pathak et al. 2017
        https://arxiv.org/pdf/1705.05363.pdf
    """

    def __init__(self, icm: IntrinsicCuriosityModule):
        self.icm = icm

    def __call__(self, experience: Experience):
        # Compute curiosity measure
        φ0, φ1 = self.icm.φ(experience.s0), self.icm.φ(experience.s1)
        φ1_hat = self.icm.forward_dynamics_model(φ0, experience.a)
        prediction_error = F.mse_loss(φ1_hat, φ1)

        # Add to experience, to avoid recomputing during BP
        experience.info['φ0'] = φ0
        experience.info['φ1'] = φ1
        experience.info['forward_model_error'] = prediction_error

        return prediction_error.detach()


def create_demons(config, env, feature_extractor, policy) -> Horde:
    # dynamics_feature_extractor = ConvBody(
    #     obs_shape=env.reset().shape,
    #     channels=(32, 32, 32, 32),
    #     kernels=(3, 3, 3, 3),
    #     strides=(2, 2, 2, 2),
    #     padding=(1, 1, 1, 1),
    #     activation=nn.ELU
    # )
    from copy import deepcopy
    dynamics_feature_extractor = deepcopy(feature_extractor)

    icm = IntrinsicCuriosityModule(
        feature=dynamics_feature_extractor,
        behavior_policy=policy,
        beta=config.get('beta', 0.2)
    )

    control_demon = AC(
        gvf=GVF(
            target_policy=Greedy(
                feature_dim=feature_extractor.feature_dim,
                action_space=env.action_space
            ),
            cumulant=CombinedCumulant({Fitness(env), Curiosity(icm)}),
            continuation=ConstantContinuation(config['gamma'])),
        behavior_policy=policy,
        feature=feature_extractor,
        criterion=F.mse_loss,
        trace_decay=config.get('trace_decay', 1)  # n-step
    )
    # TODO: pass the device with the demon
    return Horde([control_demon, icm], device('cpu'))
