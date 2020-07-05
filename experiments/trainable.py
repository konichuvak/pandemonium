from typing import Callable, Union

from ray.rllib.agents.a3c.a3c_torch_policy import A3CTorchPolicy
from ray.rllib.agents.trainer import Trainer, COMMON_CONFIG
from ray.rllib.utils import override
from ray.tune import Trainable
from ray.tune.resources import Resources

from pandemonium.agent import Agent
from pandemonium.envs import MiniGridEnv, DeepmindLabEnv
from pandemonium.networks.bodies import BaseNetwork
from pandemonium.policies import Policy

Trainer._allow_unknown_configs = True
Env = Union[MiniGridEnv, DeepmindLabEnv]


class Loop(Trainer):
    # _name = name
    _policy = A3CTorchPolicy
    _default_config = COMMON_CONFIG

    def create_encoder(self, cfg, obs_space):
        if isinstance(self.env.unwrapped, MiniGridEnv):
            from pandemonium.envs.minigrid import encoder_registry
        elif isinstance(self.env.unwrapped, DeepmindLabEnv):
            from pandemonium.envs.dm_lab import encoder_registry
        else:
            raise ValueError(f'Invalid Environment: {self.env.unwrapped}')

        encoder = encoder_registry[cfg['encoder']]
        encoder_cls = BaseNetwork.by_name(encoder['encoder_name'])
        encoder_cfg = encoder.get('encoder_cfg', {})

        obs_shape = obs_space.shape
        if len(obs_shape) == 3:
            obs_shape = (obs_shape[2], obs_shape[0], obs_shape[1])
        return encoder_cls(obs_shape=obs_shape, **encoder_cfg)

    @staticmethod
    def create_policy(cfg, feature_dim, action_space):
        policy_cls = Policy.by_name(cfg['policy_name'])
        policy_cfg = cfg.get('policy_cfg', {})
        return policy_cls(feature_dim=feature_dim,
                          action_space=action_space,
                          **policy_cfg)

    def _init(self, cfg, env_creator: Callable[[dict], Env]):
        self.env = env_creator(cfg['env_config'])
        encoder = self.create_encoder(cfg, self.env.observation_space)
        print(encoder)
        policy = self.create_policy(cfg, encoder.feature_dim, self.env.action_space)
        print(policy)
        horde = cfg['horde_fn'](cfg, self.env, encoder, policy)

        # Create a learning loop
        self.agent = Agent(encoder, policy, horde)
        self.loop = self.agent.learn(
            env=self.env,
            episodes=100000000000,
            horizon=cfg['rollout_fragment_length'],
        )

        self.hist_stats = {
            'episode_reward': list()
        }

    def _evaluate(self):
        metrics = self.config["custom_eval_function"](
            self, self.evaluation_workers)
        return metrics

    def _train(self):
        """ Perform one training iteration

        In general, a training iteration may last for multiple updates of
        the networks.
        """
        logs = next(self.loop)

        # Add histogram stats
        if 'episode_reward' in logs:
            self.hist_stats['episode_reward'].append(logs['episode_reward'])
            logs['hist_stats'] = self.hist_stats
        return logs

    @staticmethod
    def _validate_config(config):
        """ Assert any structure that config should posses """
        super()._validate_config(config)

    @classmethod
    @override(Trainable)
    def default_resource_request(cls, config):
        cf = dict(cls._default_config, **config)
        Trainer._validate_config(cf)
        num_workers = cf["num_workers"] + cf["evaluation_num_workers"]
        # TODO(ekl): add custom resources here once tune supports them
        return Resources(
            cpu=cf["num_cpus_for_driver"],
            gpu=cf["num_gpus"],
            memory=cf["memory"],
            object_store_memory=cf["object_store_memory"],
            extra_cpu=0,  # NOTE: removing extra CPU's since sampling is synced
            extra_gpu=cf["num_gpus_per_worker"] * num_workers,
            extra_memory=cf["memory_per_worker"] * num_workers,
            extra_object_store_memory=cf["object_store_memory_per_worker"] *
                                      num_workers)

    @property
    def _name(self):
        return self.__class__.__name__

    # def _save(self, tmp_checkpoint_dir):
    #     # tmp_checkpoint_dir = PARAMETER_DIR / f'{episode}.pt'
    #     torch.save(self.agent.horde.state_dict(), tmp_checkpoint_dir)
    #     return tmp_checkpoint_dir
    #
    # def _restore(self, checkpoint):
    #     # experiment_id = '2020-03-19 14:58:10'
    #     weight_name = '1600.pt'
    #     self.agent.horde.load_state_dict(
    #         state_dict=torch.load(
    #             f=EXPERIMENT_DIR / self._experiment_id / 'weights' / weight_name,
    #             map_location=device,
    #         ),
    #         strict=True
    #     )
