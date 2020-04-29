import torch
from ray.rllib.agents.trainer import Trainer, COMMON_CONFIG
from ray.rllib.utils import override
from ray.tune import Trainable
from ray.tune.resources import Resources

from pandemonium.agent import Agent
from pandemonium.networks.bodies import BaseNetwork
from pandemonium.policies import Policy

Trainer._allow_unknown_configs = True


class Loop(Trainer):
    # _name = name
    # _policy = default_policy
    _default_config = COMMON_CONFIG

    def _init(self, cfg, env_creator):
        # Set up the environment
        self.env = env_creator(cfg['env_config'])

        # Set up the feature extractor
        net_cls = BaseNetwork.by_name(cfg['feature_name'])
        obs_shape = self.env.reset().shape
        if isinstance(obs_shape, torch.Size):
            obs_shape = obs_shape[1:]  # discard batch dimension
        feature_extractor = net_cls(obs_shape=obs_shape, **cfg['feature_cfg'])

        # Set up policy network
        policy_cls = Policy.by_name(cfg['policy_name'])
        policy = policy_cls(feature_dim=feature_extractor.feature_dim,
                            action_space=self.env.action_space,
                            **cfg['policy_cfg'])

        # Set up Optimizer a.k.a. Horde
        horde = cfg['horde_fn'](cfg, self.env, feature_extractor, policy)

        # Create a learning loop
        agent = Agent(feature_extractor, policy, horde)
        self.loop = agent.learn(
            env=self.env,
            episodes=10000,
            update_freq=cfg['rollout_fragment_length'],
        )

    def _train(self):
        """ Perform one training iteration

        A training iteration may last for multiple updates of the networks.
        """
        logs = next(self.loop)
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
