from typing import Dict

from experiments.trainable import Loop
from pandemonium.demons import ControlDemon, PredictionDemon
from pandemonium.demons.control import CategoricalQ
from pandemonium.envs.minigrid import MinigridDisplay


def eval_fn(trainer: Loop, eval_workers) -> Dict:
    """

    Called every `evaluation_interval` to run the current version of the
    agent in the the evaluation environment for one episode.

    Works for envs with fairly small, enumerable state space like gridworlds.

    Parameters
    ----------
    trainer
    eval_workers

    Returns
    -------

    """
    cfg = trainer.config['evaluation_config']
    env = cfg['eval_env'](trainer.config['env_config'])

    display = MinigridDisplay(env)

    iteration = trainer.iteration

    # Visualize value functions of each demon
    for demon in trainer.agent.horde.demons.values():

        if isinstance(demon, ControlDemon):
            # fig = display.plot_option_values(
            #     figure_name=f'iteration {iteration}',
            #     demon=demon,
            # )
            fig = display.plot_option_values_separate(
                figure_name=f'iteration {iteration}',
                demon=demon,
            )
            display.save_figure(fig, f'{trainer.logdir}/{iteration}_qf')

            if isinstance(demon, CategoricalQ) and hasattr(demon, 'num_atoms'):
                fig = display.plot_option_value_distributions(
                    figure_name=f'iteration {iteration}',
                    demon=demon,
                )
                print(f'saving @ {trainer.logdir}/{iteration}_zf')
                display.save_figure(fig, f'{trainer.logdir}/{iteration}_zf')

        elif isinstance(demon, PredictionDemon):
            pass

    return {'dummy': None}
