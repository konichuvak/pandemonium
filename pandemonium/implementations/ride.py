from collections import defaultdict

import torch
import torch.nn.functional as F
from pandemonium import Horde, GVF
from pandemonium.continuations import ConstantContinuation
from pandemonium.cumulants import CombinedCumulant, Fitness
from pandemonium.experience import Experience
from pandemonium.implementations.a2c import AC
from pandemonium.implementations.icm import Curiosity, ICM
from pandemonium.policies import Greedy
from pandemonium.utilities.utilities import get_all_members


class ImpactDrivenCuriosity(Curiosity):
    r""" A slight upgrade of curiosity-based intrinsic reward.

    Uses the __actual__ next state features $\phi(s_{t+1})$ instead of the
    __predicted__ next state features $\hat{\phi}(s_{t+1})$ as a target in
    L2 loss. Moreover, the loss is normalized by the episodic feature
    pseudo-count, to avoid alternating between very different latents in a loop.
    """

    def __init__(self, icm: ICM):
        super().__init__(icm=icm)
        self.counter = defaultdict(lambda: 1.)

    def __call__(self, experience: Experience):
        super().__call__(experience)
        counts = self.update_visitation_counts(experience)
        φ0, φ1 = experience.info['φ0'], experience.info['φ1']
        intrinsic_reward = torch.norm(φ1 - φ0, p=2, dim=1) / counts
        experience.info['intrinsic_reward'] = intrinsic_reward.mean().item()
        return intrinsic_reward.detach()

    def update_visitation_counts(self, experience: Experience) -> torch.Tensor:

        # Save and reset episodic counters
        if any(experience.done) and self.counter:
            counts = list(self.counter.values())
            experience.info['max_visitations'] = max(counts)
            experience.info['mean_visitations'] = sum(counts) / len(counts)
            experience.info['min_visitations'] = min(counts)
            self.counter = defaultdict(lambda: 1.)

        # Update the counts using states in the trajectory
        counts = list()
        for i in range(len(experience)):
            s = experience.info['agent_pos'][i]
            # s = tuple(experience.s1[i].view(-1).tolist())
            self.counter[s] += 1
            counts.append(self.counter[s])
        return torch.tensor(counts, device=experience.x1.device)


def create_horde(config, env, feature_extractor, policy) -> Horde:
    demons = list()

    cumulant = Fitness(env)

    if config.get('icm_weight', 1.):
        from copy import deepcopy
        dynamics_feature_extractor = deepcopy(feature_extractor)

        icm = ICM(
            feature=dynamics_feature_extractor,
            behavior_policy=policy,
            beta=config.get('beta', 0.2)
        )
        demons.append(icm)
        cumulant = CombinedCumulant(
            cumulants={Fitness(env), ImpactDrivenCuriosity(icm)},
            weights=torch.tensor([config.get('er_weight', 1.),
                                  config.get('ir_weight', 0.1)]),
        )

    # Note that order matters! When we iterate over a collection of demons
    #   in horde.py, we will start with the last demon, ICM, which
    #   will share information required for computing intrinsic reward in the
    #   main agent's learning loop.
    demons.insert(0, AC(
        gvf=GVF(
            target_policy=Greedy(
                feature_dim=feature_extractor.feature_dim,
                action_space=env.action_space
            ),
            cumulant=cumulant,
            continuation=ConstantContinuation(config['gamma'])),
        behavior_policy=policy,
        feature=feature_extractor,
        criterion=F.mse_loss,
        trace_decay=config.get('trace_decay', 1)  # n-step
    ))
    weights = [
        config.get('ac_weight', 1.),
        config.get('icm_weight', 1.)
    ]

    device = torch.device('cpu')
    demon_weights = torch.tensor(
        data=[w for w in weights if w],
        dtype=torch.float
    ).to(device)

    return Horde(
        demons=demons,
        device=device,
        aggregation_fn=lambda losses: demon_weights.dot(losses),
        to_transitions=True,
    )


__all__ = get_all_members(__name__)
