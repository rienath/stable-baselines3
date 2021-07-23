import os
from typing import Any, Callable, Dict, Optional, Type, Union

import gym
import retro

from stable_baselines3.common.atari_wrappers import AtariWrapper, EpisodicLifeEnv, ClipRewardEnv, StochasticFrameSkip, WarpFrame, FireResetEnv, RetroSound
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv


def unwrap_wrapper(env: gym.Env, wrapper_class: Type[gym.Wrapper]) -> Optional[gym.Wrapper]:
    """
    Retrieve a ``VecEnvWrapper`` object by recursively searching.

    :param env: Environment to unwrap
    :param wrapper_class: Wrapper to look for
    :return: Environment unwrapped till ``wrapper_class`` if it has been wrapped with it
    """
    env_tmp = env
    while isinstance(env_tmp, gym.Wrapper):
        if isinstance(env_tmp, wrapper_class):
            return env_tmp
        env_tmp = env_tmp.env
    return None


def is_wrapped(env: Type[gym.Env], wrapper_class: Type[gym.Wrapper]) -> bool:
    """
    Check if a given environment has been wrapped with a given wrapper.

    :param env: Environment to check
    :param wrapper_class: Wrapper class to look for
    :return: True if environment has been wrapped with ``wrapper_class``.
    """
    return unwrap_wrapper(env, wrapper_class) is not None


def make_vec_env(
    env_id: Union[str, Type[gym.Env]],
    n_envs: int = 1,
    seed: Optional[int] = None,
    start_index: int = 0,
    monitor_dir: Optional[str] = None,
    wrapper_class: Optional[Callable[[gym.Env], gym.Env]] = None,
    retro: bool = False,
    retro_obs_type: Optional[retro.Observations] = retro.Observations.IMAGE,
    env_kwargs: Optional[Dict[str, Any]] = None,
    vec_env_cls: Optional[Type[Union[DummyVecEnv, SubprocVecEnv]]] = None,
    vec_env_kwargs: Optional[Dict[str, Any]] = None,
    monitor_kwargs: Optional[Dict[str, Any]] = None,
    wrapper_kwargs: Optional[Dict[str, Any]] = None,
) -> VecEnv:
    """
    Create a wrapped, monitored ``VecEnv``.
    By default it uses a ``DummyVecEnv`` which is usually faster
    than a ``SubprocVecEnv``.

    :param env_id: the environment ID or the environment class
    :param n_envs: the number of environments you wish to have in parallel
    :param seed: the initial seed for the random number generator
    :param start_index: start rank index
    :param monitor_dir: Path to a folder where the monitor files will be saved.
        If None, no file will be written, however, the env will still be wrapped
        in a Monitor wrapper to provide additional information about training.
    :param wrapper_class: Additional wrapper to use on the environment.
        This can also be a function with single argument that wraps the environment in many things.
    :param env_kwargs: Optional keyword argument to pass to the env constructor
    :param vec_env_cls: A custom ``VecEnv`` class constructor. Default: None.
    :param vec_env_kwargs: Keyword arguments to pass to the ``VecEnv`` class constructor.
    :param monitor_kwargs: Keyword arguments to pass to the ``Monitor`` class constructor.
    :param wrapper_kwargs: Keyword arguments to pass to the ``Wrapper`` class constructor.
    :return: The wrapped environment
    """
    env_kwargs = {} if env_kwargs is None else env_kwargs
    vec_env_kwargs = {} if vec_env_kwargs is None else vec_env_kwargs
    monitor_kwargs = {} if monitor_kwargs is None else monitor_kwargs
    wrapper_kwargs = {} if wrapper_kwargs is None else wrapper_kwargs

    def make_env(rank):
        def _init():
            if isinstance(env_id, str):
                env = gym.make(env_id, **env_kwargs) if not retro_bool else retro.make(
                    game=env_id, 
                    use_restricted_actions=retro.Actions.DISCRETE, 
                    obs_type=retro_obs_type)
            else:
                env = env_id(**env_kwargs)
            if seed is not None:
                env.seed(seed + rank)
                env.action_space.seed(seed + rank)
            # Wrap the env in a Monitor wrapper
            # to have additional training information
            monitor_path = os.path.join(monitor_dir, str(rank)) if monitor_dir is not None else None
            # Create the monitor folder if needed
            if monitor_path is not None:
                os.makedirs(monitor_dir, exist_ok=True)
            env = Monitor(env, filename=monitor_path, **monitor_kwargs)
            # Optionally, wrap the environment with the provided wrapper
            if wrapper_class is not None:
                env = wrapper_class(env, **wrapper_kwargs)
            return env

        return _init

    # No custom VecEnv is passed
    if vec_env_cls is None:
        # Default: use a DummyVecEnv
        vec_env_cls = DummyVecEnv

    return vec_env_cls([make_env(i + start_index) for i in range(n_envs)], **vec_env_kwargs)


def make_atari_env(
    env_id: Union[str, Type[gym.Env]],
    n_envs: int = 1,
    seed: Optional[int] = None,
    start_index: int = 0,
    monitor_dir: Optional[str] = None,
    wrapper_kwargs: Optional[Dict[str, Any]] = None,
    env_kwargs: Optional[Dict[str, Any]] = None,
    vec_env_cls: Optional[Union[DummyVecEnv, SubprocVecEnv]] = None,
    vec_env_kwargs: Optional[Dict[str, Any]] = None,
    monitor_kwargs: Optional[Dict[str, Any]] = None,
) -> VecEnv:
    """
    Create a wrapped, monitored VecEnv for Atari.
    It is a wrapper around ``make_vec_env`` that includes common preprocessing for Atari games.

    :param env_id: the environment ID or the environment class
    :param n_envs: the number of environments you wish to have in parallel
    :param seed: the initial seed for the random number generator
    :param start_index: start rank index
    :param monitor_dir: Path to a folder where the monitor files will be saved.
        If None, no file will be written, however, the env will still be wrapped
        in a Monitor wrapper to provide additional information about training.
    :param wrapper_kwargs: Optional keyword argument to pass to the ``AtariWrapper``
    :param env_kwargs: Optional keyword argument to pass to the env constructor
    :param vec_env_cls: A custom ``VecEnv`` class constructor. Default: None.
    :param vec_env_kwargs: Keyword arguments to pass to the ``VecEnv`` class constructor.
    :param monitor_kwargs: Keyword arguments to pass to the ``Monitor`` class constructor.
    :return: The wrapped environment
    """
    if wrapper_kwargs is None:
        wrapper_kwargs = {}

    def atari_wrapper(env: gym.Env) -> gym.Env:
        env = AtariWrapper(env, **wrapper_kwargs)
        return env

    return make_vec_env(
        env_id,
        n_envs=n_envs,
        seed=seed,
        start_index=start_index,
        monitor_dir=monitor_dir,
        wrapper_class=atari_wrapper,
        env_kwargs=env_kwargs,
        vec_env_cls=vec_env_cls,
        vec_env_kwargs=vec_env_kwargs,
        monitor_kwargs=monitor_kwargs,
    )

def make_retro_env(
    env_id: Union[str, Type[gym.Env]],
    seed: Optional[int] = None,
    start_index: int = 0,
    monitor_dir: Optional[str] = None,
    monitor_kwargs: Optional[Dict[str, Any]] = None,
    wrapper_kwargs: Optional[Dict[str, Any]] = None,
    env_kwargs: Optional[Dict[str, Any]] = None,
    screen_size: int = 84,
    terminal_on_life_loss: bool = True,
    clip_reward: bool = True,
    fire_at_the_start: bool = False,
    audio: bool = False
) -> VecEnv:
    """
    Create a wrapped, monitored VecEnv for Gym Retro.
    It is a wrapper around ``make_vec_env`` that includes common preprocessing for Retro Atari games.
    """

    env_kwargs = {} if env_kwargs is None else env_kwargs
    monitor_kwargs = {} if monitor_kwargs is None else monitor_kwargs
    wrapper_kwargs = {} if wrapper_kwargs is None else wrapper_kwargs

    def parse_env_name(env_id):
        """
        Gym Retro uses different env_id's to standard Gym. 
        This method will parse the Gym environment id and return all the information needed to 
        make a desired Gym Retro environment.
        """
        gym_env = gym.envs.spec(env_id)
        env_id = gym_env.id
        max_episode_steps = gym_env.max_episode_steps
        repeat_action_probability = 0
        obs_type=retro.Observations.IMAGE

        # The environments by default choose frameskip number randomly from 2-4 at every step.
        # That is why we distinguish between frameskip_min and frameskip_max.
        frameskip_min, frameskip_max = 2, 4
        # Space Invaders is an exception.
        if env_id == 'space_invaders':
            frameskip_min, frameskip_max = 3, 3
        else:
            frameskip_min, frameskip_max = 4, 4

        # The longest form of env_id there can be is {env_id}-{ram}{Deterministic/NoFrameskip}-{v{0,4}}.
        # ram and Deterministic/NoFrameskip are optional, so
        # first let's check for {Deterministic/NoFrameskip}-{v{0,4}} and delete them 
        # and update needed parameters.
        # v0 environments have 25% chance of repeating previous action, v4 have 0%.

        if 'Deterministic-v0' in env_id:
            env_id = env_id[:-16]
            repeat_action_probability = 0.25
            frameskip_min, frameskip_max = 4, 4
        
        elif 'Deterministic-v4' in env_id:
            env_id = env_id[:-16]
            frameskip_min, frameskip_max = 4, 4

        elif 'NoFrameskip-v0' in env_id:
            env_id = env_id[:-14]
            repeat_action_probability = 0.25
            frameskip_min, frameskip_max = 1, 1

        elif 'NoFrameskip-v4' in env_id:
            env_id = env_id[:-14]
            frameskip_min, frameskip_max = 1, 1

        elif 'v0' in env_id:
            env_id = env_id[:-3]
            repeat_action_probability = 0.25

        elif 'v4' in env_id:
            env_id = env_id[:-3]

        # Raise error if none work
        else:
            raise('Such environment name does not exist.')

        if 'ram' in env_id:
            env_id = env_id[:-4]
            obs_type=retro.Observations.RAM

        # Since we only support Atari games, 
        # env will have -Atari2600 suffix, which Retro requires.
        env_id+='-Atari2600'

        return env_id, obs_type, repeat_action_probability, max_episode_steps, frameskip_min, frameskip_max

    env_id, obs_type, repeat_action_probability, max_episode_steps, frameskip_min, frameskip_max = parse_env_name(env_id)

    # The actions must be discrete, but are not by default.
    env = retro.make(game=env_id, use_restricted_actions=retro.Actions.DISCRETE, obs_type=obs_type)

    if seed is not None:
        env.seed(seed)
        env.action_space.seed(seed)

    # Wrap the env in a Monitor wrapper
    # to have additional training information
    monitor_path = os.path.join(monitor_dir) if monitor_dir is not None else None
    # Create the monitor folder if needed
    if monitor_path is not None:
        os.makedirs(monitor_dir, exist_ok=True)
    env = Monitor(env, filename=monitor_path, **monitor_kwargs)

    # Add wrappers to skip frames and limit env time.
    env = StochasticFrameSkip(env, frameskip_min, frameskip_max, repeat_action_probability, audio)
    env = WarpFrame(env, width=84, height=84)
    if audio:
        env = RetroSound(env, frameskip_max)
    env = gym.wrappers.TimeLimit(env, max_episode_steps)
    if fire_at_the_start:
        env = FireResetEnv(env)

    # Does not work with retro
    #if terminal_on_life_loss:
    #    env = EpisodicLifeEnv(env)
    if clip_reward:
        env = ClipRewardEnv(env)
    
    return env
