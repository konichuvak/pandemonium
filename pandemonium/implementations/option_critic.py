import torch

from pandemonium.demons import Loss
from pandemonium.demons.ac import ActorCritic
from pandemonium.implementations.rainbow import DQN
from pandemonium.experience import ER, Trajectory, Transitions
from pandemonium.policies import HierarchicalPolicy


class OC(ActorCritic, DQN):
    """ DQN style Option-Critic architecture """

    def __init__(self, actor: HierarchicalPolicy, **kwargs):
        super().__init__(
            output_dim=len(actor.option_space),
            actor=actor,
            # TODO: make a sequential replay that samples last
            #  BATCH_SIZE transitions in order
            replay_buffer=ER(size=0, batch_size=0),
            **kwargs
        )

        # Collect parameters of all the options into one computational graph
        # TODO: instead of manually collecting we can make HierarchicalPolicy
        #  and OptionSpace subclass nn.Module
        for idx, o in self.μ.option_space.options.items():
            for k, param in o.policy.named_parameters(f'option_{idx}'):
                self.register_parameter(k.replace('.', '_'), param)
            for k, param in o.continuation.named_parameters(f'option_{idx}'):
                self.register_parameter(k.replace('.', '_'), param)

    def learn(self, transitions: Transitions):
        self.update_counter += 1
        self.sync_target()
        trajectory = Trajectory.from_transitions(transitions)
        return self.delta(trajectory)

    def delta(self, traj: Trajectory) -> Loss:
        η = 0.001
        ω = traj.info['option'].unsqueeze(1)
        β = traj.info['beta']
        π = traj.info['action_dist']

        # ------------------------------------
        # Value gradient
        # TODO: should we pass through feature generator again?
        #   if yes, just use: value_loss, info = super(DQN, self).delta(traj)
        #   with actions replaced by options
        x = traj.x0
        values = self.predict(x)[torch.arange(x.size(0)), ω]
        targets = self.n_step_target(traj).detach()
        value_loss = self.criterion(values, targets)
        values = values.detach()

        # ------------------------------------
        # Policy gradient
        # TODO: re-compute targets with current net instead of target net?
        #  see PLB p. 116
        advantage = targets - values
        log_probs = torch.cat([pi.log_prob(a) for pi, a in zip(π, traj.a)])
        policy_grad = (-log_probs * advantage).mean(0)

        entropy = torch.cat([pi.entropy() for pi in π])
        entropy_reg = [o.policy.β for o in self.μ.option_space[ω.squeeze(1)]]
        entropy_reg = torch.tensor(entropy_reg, device=entropy.device)
        entropy_loss = (entropy_reg * entropy).mean(0)

        policy_loss = policy_grad - entropy_loss

        # ------------------------------------
        # Termination gradient
        termination_advantage = values - values.max()
        beta_loss = (β * (termination_advantage + η)).mean()

        loss = policy_loss + value_loss + beta_loss
        return loss, {
            'policy_grad': policy_loss.item(),
            'entropy': entropy_loss.item(),
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'beta_loss': beta_loss.item(),
            'loss': loss.item(),
        }

    def n_step_target(self, traj: Trajectory):
        ω = traj.info['option'].unsqueeze(1)
        β = traj.info['beta']
        γ = self.gvf.continuation(traj)
        z = self.gvf.cumulant(traj)
        q = self.target_avf(traj.x1[-1])

        targets = torch.empty_like(z, dtype=torch.float)
        u = β[-1] * q[ω[-1]] + (1 - β[-1]) * q.max()
        for i in range(len(traj) - 1, -1, -1):
            u = targets[i] = z[i] + γ[i] * u
        return targets
