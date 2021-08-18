import gym
import numpy as np
from gym import spaces
from functools import lru_cache

try:
    import cv2  # pytype:disable=import-error

    cv2.ocl.setUseOpenCL(False)
except ImportError:
    cv2 = None

from stable_baselines3.common.type_aliases import GymObs, GymStepReturn


class NoopResetEnv(gym.Wrapper):
    """
    Sample initial states by taking random number of no-ops on reset (Mnih et al., 2015).
    No-op is assumed to be action 0.

    :param env: the environment to wrap
    :param noop_max: the maximum value of no-ops to run
    """

    def __init__(self, env: gym.Env, noop_max: int = 30, retro: bool = False):
        gym.Wrapper.__init__(self, env)
        self.noop_max = noop_max
        self.override_num_noops = None
        self.noop_action = 0
        if not retro:
            assert env.unwrapped.get_action_meanings()[0] == "NOOP"

    def reset(self, **kwargs) -> np.ndarray:
        self.env.reset(**kwargs)
        if self.override_num_noops is not None:
            noops = self.override_num_noops
        else:
            noops = self.unwrapped.np_random.randint(1, self.noop_max + 1)
        assert noops > 0
        obs = np.zeros(0)
        for _ in range(noops):
            obs, _, done, _ = self.env.step(self.noop_action)
            if done:
                obs = self.env.reset(**kwargs)
        return obs


class FireResetEnv(gym.Wrapper):
    """
    Take action on reset for environments that are fixed until firing.

    :param env: the environment to wrap
    """

    def __init__(self, env: gym.Env, retro: bool = False):
        gym.Wrapper.__init__(self, env)
        # Retro does not have action meanings
        if not retro:
            assert env.unwrapped.get_action_meanings()[1] == "FIRE"
            assert len(env.unwrapped.get_action_meanings()) >= 3

    def reset(self, **kwargs) -> np.ndarray:
        self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(1)
        if done:
            self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(2)
        if done:
            self.env.reset(**kwargs)
        return obs


class EpisodicLifeEnv(gym.Wrapper):
    """
    Make end-of-life == end-of-episode, but only reset on true game over.
    Done by DeepMind for the DQN and co. since it helps value estimation.

    :param env: the environment to wrap
    """

    def __init__(self, env: gym.Env, retro: bool = False):
        gym.Wrapper.__init__(self, env)
        self.lives = 0
        self.was_real_done = True
        # Gym and retro have different ways of getting lives.
        if retro:
            self.get_lives = lambda: self.env.unwrapped.data['lives']
        else:
            self.get_lives = lambda: self.env.unwrapped.ale.lives()


    def step(self, action: int) -> GymStepReturn:
        obs, reward, done, info = self.env.step(action)
        self.was_real_done = done
        # check current lives, make loss of life terminal,
        # then update lives to handle bonus lives
        lives = self.get_lives()
        if 0 < lives < self.lives:
            # for Qbert sometimes we stay in lives == 0 condtion for a few frames
            # so its important to keep lives > 0, so that we only reset once
            # the environment advertises done.
            done = True
        self.lives = lives
        return obs, reward, done, info

    def reset(self, **kwargs) -> np.ndarray:
        """
        Calls the Gym environment reset, only when lives are exhausted.
        This way all states are still reachable even though lives are episodic,
        and the learner need not know about any of this behind-the-scenes.

        :param kwargs: Extra keywords passed to env.reset() call
        :return: the first observation of the environment
        """
        if self.was_real_done:
            obs = self.env.reset(**kwargs)
        else:
            # no-op step to advance from terminal/lost life state
            obs, _, _, _ = self.env.step(0)
        self.lives = self.get_lives()
        return obs

class MaxAndSkipEnv(gym.Wrapper):
    """
    Return only every ``skip``-th frame (frameskipping)

    :param env: the environment
    :param skip: number of ``skip``-th frame
    """

    def __init__(self, env: gym.Env, skip: int = 4):
        gym.Wrapper.__init__(self, env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = np.zeros((2,) + env.observation_space.shape, dtype=env.observation_space.dtype)
        self._skip = skip

    def step(self, action: int) -> GymStepReturn:
        """
        Step the environment with the given action
        Repeat action, sum reward, and max over last observations.

        :param action: the action
        :return: observation, reward, done, information
        """
        total_reward = 0.0
        done = None
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            if i == self._skip - 2:
                self._obs_buffer[0] = obs
            if i == self._skip - 1:
                self._obs_buffer[1] = obs
            total_reward += reward
            if done:
                break
        # Note that the observation on the done=True frame
        # doesn't matter
        max_frame = self._obs_buffer.max(axis=0)

        return max_frame, total_reward, done, info

    def reset(self, **kwargs) -> GymObs:
        return self.env.reset(**kwargs)


class ClipRewardEnv(gym.RewardWrapper):
    """
    Clips the reward to {+1, 0, -1} by its sign.

    :param env: the environment
    """

    def __init__(self, env: gym.Env):
        gym.RewardWrapper.__init__(self, env)

    def reward(self, reward: float) -> float:
        """
        Bin reward to {+1, 0, -1} by its sign.

        :param reward:
        :return:
        """
        return np.sign(reward)


class WarpFrame(gym.ObservationWrapper):
    """
    Convert to grayscale and warp frames to 84x84 (default)
    as done in the Nature paper and later work.

    :param env: the environment
    :param width:
    :param height:
    """

    def __init__(self, env: gym.Env, width: int = 84, height: int = 84):
        gym.ObservationWrapper.__init__(self, env)
        self.width = width
        self.height = height
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(self.height, self.width, 1), dtype=env.observation_space.dtype
        )

    def observation(self, frame: np.ndarray) -> np.ndarray:
        """
        returns the current observation from a frame

        :param frame: environment frame
        :return: the observation
        """
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        return frame[:, :, None]


class AtariWrapper(gym.Wrapper):
    """
    Atari 2600 preprocessings

    Specifically:

    * NoopReset: obtain initial state by taking random number of no-ops on reset.
    * Frame skipping: 4 by default
    * Max-pooling: most recent two observations
    * Termination signal when a life is lost.
    * Resize to a square image: 84x84 by default
    * Grayscale observation
    * Clip reward to {-1, 0, 1}

    :param env: gym environment
    :param noop_max: max number of no-ops
    :param frame_skip: the frequency at which the agent experiences the game.
    :param screen_size: resize Atari frame
    :param terminal_on_life_loss: if True, then step() returns done=True whenever a life is lost.
    :param clip_reward: If True (default), the reward is clip to {-1, 0, 1} depending on its sign.
    """

    def __init__(
        self,
        env: gym.Env,
        noop_max: int = 30,
        frame_skip: int = 4,
        screen_size: int = 84,
        terminal_on_life_loss: bool = True,
        clip_reward: bool = True,
    ):
        env = NoopResetEnv(env, noop_max=noop_max)
        env = MaxAndSkipEnv(env, skip=frame_skip)
        if terminal_on_life_loss:
            env = EpisodicLifeEnv(env)
        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        env = WarpFrame(env, width=screen_size, height=screen_size)
        if clip_reward:
            env = ClipRewardEnv(env)

        super(AtariWrapper, self).__init__(env)


class StochasticFrameskip(gym.Wrapper):
    """
    Should be used before other wrappers to make sure that no unnecessary computation is done with skipped frames.
    The only exception is WarpFrame, which should happen be used before or it will prevent tight
    StochasticFrameskip and RetroSound integration.
    Sticky actions as in Machado et al., 2017, random frameskips as in Brockman et al., 2016.
    """
    def __init__(self, env, step_min, step_max, stickprob, audio=False):
        gym.Wrapper.__init__(self, env)
        self.step_min = step_min
        self.step_max = step_max
        self.stickprob = stickprob
        self.curac = None
        self.rng = np.random.RandomState()
        self.supports_want_render = hasattr(env, "supports_want_render")
        self.audio = audio

    def reset(self, **kwargs):
        self.curac = None
        return self.env.reset(**kwargs)

    def step(self, ac):
        done = False
        totrew = 0
        # Indicates whether the actual action has been used already. False while sticky actions are taking place
        real_action_used = False
        if self.audio: audio_buffer = list()
        # Choose random frameskip. Will stay constant if min and max are equal.
        n = self.rng.randint(self.step_min, self.step_max + 1)
        for i in range(n):
            # First step after reset, use action
            if self.curac is None:
                self.curac = ac
            # Steps not after reset.
            # Take previous action if real action has not been used yet with stickprob probability.
            elif not real_action_used:
                if self.rng.rand() > self.stickprob:
                    self.curac = ac
                    real_action_used = True
            if self.supports_want_render and i<self.n-1:
                ob, rew, done, info = self.env.step(self.curac, want_render=False)
            else:
                ob, rew, done, info = self.env.step(self.curac)
            totrew += rew
            if self.audio: audio_buffer.append(self.env.em.get_audio())
            if done: break
        if self.audio: 
            # Instead of having 3D array with frame number as 3rd dimension,
            # reshape into 2D array with continuous multiframe audio
            audio_buffer = np.array(audio_buffer)
            audio_buffer = audio_buffer.reshape(-1,2)
            return ob, totrew, done, info, audio_buffer
        else: 
            return ob, totrew, done, info

    def seed(self, s):
        self.rng.seed(s)
        self.env.seed(s)


class RetroSound(gym.Wrapper):
    """
    Modify observation space and add audio to it on each step.

    :param env: gym environment
    :param frameskip: the maximum frameskip of the environment. 1 is equivalent to no frameskipping.
    """

    def __init__(self, env, frameskip=1):
        gym.Wrapper.__init__(self, env)
        current_obs_space = self.observation_space

        assert frameskip > 0
        # Find the length of momentary audio and multiply it by the number of frames skipped
        # as we need audio from the skipped frames too.
        self.single_audio_len = len(self.env.em.get_audio())
        self.frameskip = frameskip
        self.audio_len = self.single_audio_len * self.frameskip

        audio_shape = [self.audio_len,2]
        sound_high = [[32767,32767]] * self.audio_len
        sound_low = [[-32767,-32767]] * self.audio_len

        # Create new observation space
        if isinstance(self.observation_space, spaces.Dict):
            new_dict = {}
            for space_name, space in self.observation_space.spaces.items():
                new_dict[space_name] = space
            new_dict['sound'] = gym.spaces.Box(
                low=np.array(sound_low, dtype=np.int16), high=np.array(sound_high, dtype=np.int16),
            )
            self.observation_space = gym.spaces.Dict(new_dict) 
        else:
            self.observation_space = gym.spaces.Dict({
                'obs': current_obs_space,
                'sound': gym.spaces.Box(
                    low=np.array(sound_low, dtype=np.int16), high=np.array(sound_high, dtype=np.int16),
                ),
            })

    def __process_audio(self, audio):
        # If the audio is smaller than the observation space and frameskip exists, fill the missing parts with 0's.
        # This can happen if minimum frameskip is 2 and maximum is 4. Then if random frameskip is chosen
        # at step n, the audio can be made for 2 frameskips even though observation space is designed for more.
        if self.observation_space['sound'].shape[0] > len(audio) and self.frameskip > 1:
            # Pad the audio with 0's
            audio_shape = np.shape(audio)
            padded_audio = np.zeros(self.observation_space['sound'].shape)
            padded_audio[:audio_shape[0],:audio_shape[1]] = audio
            audio = padded_audio
        elif self.observation_space['sound'].shape[0] > len(audio):
            raise Exception('Length of audio should not be smaller than the audio in observation space')
            audio = np.zeros(self.observation_space['sound'].shape)
        elif self.observation_space['sound'].shape[0] < len(audio):
            raise Exception('Length of audio should not be larger than the audio in observation space')
            audio = np.zeros(self.observation_space['sound'].shape)
        return audio

    def reset(self, **kwargs):
        base_observation = self.env.reset(**kwargs)
        audio = self.env.em.get_audio()
        audio = self.__process_audio(audio)

        if isinstance(base_observation, dict):
            base_observation['sound'] = audio
            return base_observation
        else:
            obs_dict = {
                'obs':base_observation,
                'sound':audio
            }
            return obs_dict

    def step(self, action):
        # If there is a frameskip, the StochasticFrameskip wrapper will return the audio out of the buffer
        if self.frameskip > 1:
            obs, rew, done, info, audio = self.env.step(action)
        else: 
            obs, rew, done, info = self.env.step(action)
            audio = self.env.em.get_audio()

        audio = self.__process_audio(audio)

        if isinstance(obs, dict):
            obs['sound'] = audio
            return obs, rew, done, info

        else:
            obs_dict = {
                'obs':obs,
                'sound':audio
            }
            return obs_dict, rew, done, info


class FFTWrapper(gym.Wrapper):
    """
    Only works with dual channel audio.
    """
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)

        assert isinstance(self.observation_space, spaces.Dict)
        assert 'sound' in self.observation_space.spaces 
        self.audio_len = self.observation_space['sound'].shape[0]
        self.fft_len = self.audio_len // 2

        # The maximum possible number in fft is ln(32767*audio_len*frameskip). 
        # Since audio_len in retro is 524, we can try the calculation with the smallest frameskip (1) 
        # and an insanely large framskip that we will never encounter (100).
        # We will get 16.7 and 21.3. Therefore, it is a good guess that we will never go beyond 21.3.
        # The minimum number we can possibly have is ln(1e-5), which is -11.5... Round it further down.
        fft_high = [[21.3, 21.3]] * self.fft_len
        fft_low = [[-11.6, -11.6]] * self.fft_len

        # Make new observation_space
        new_dict = {}
        for space_name, space in self.observation_space.spaces.items():
            new_dict[space_name] = space
        new_dict['sound'] = gym.spaces.Box(
            low=np.array(fft_low, dtype=np.int16), high=np.array(fft_high, dtype=np.int16),
        )
        self.observation_space = gym.spaces.Dict(new_dict) 

    @lru_cache(maxsize=128)
    def __hamming(self, num):
        return np.hamming(num)

    def __fft(self, audio):
        """
        FFT 1D vector.
        """
        audio_len = len(audio)
        audio = self.__hamming(audio_len) * audio
        fourier = np.fft.fft(audio)
        # Take only the magnitudes of different frequencies
        magnitudes = np.abs(fourier)
        # Last half of the magnitude are mirror of first half
        magnitudes = magnitudes[:audio_len // 2]
        # Take logarithm, as these magnitudes may be very large, and networks do not enjoy large values
        log_magnitudes = np.log(magnitudes + 1e-5)
        return log_magnitudes

    def __fft_two_channels(self, audio):
        """
        FFT the 2 channel audio.
        """
        # Split the signal into left and right ears
        transposed_audio = np.transpose(audio)
        ear_one = transposed_audio[0]
        ear_two = transposed_audio[1]
        # FFT on both ears
        ear_one_fft = self.__fft(ear_one)
        ear_two_fft = self.__fft(ear_two)
        # Get it to initial format
        fft_audio = np.transpose([ear_one_fft, ear_two_fft])
        return fft_audio


    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        fft_audio = self.__fft_two_channels(obs['sound'])
        obs['sound'] = fft_audio
        return obs

    def step(self, ac):
        obs, rew, done, info = self.env.step(ac)
        fft_audio = self.__fft_two_channels(obs['sound'])
        obs['sound'] = fft_audio
        return obs, rew, done, info


class Discretizer(gym.ActionWrapper):
    """
    Wrap a gym environment and make it use discrete actions.
    :param buttons: ordered list of buttons, corresponding to each dimension of the MultiBinary action space
    :param combos: ordered list of lists of valid button combinations
    """

    def __init__(self, env, buttons, combos):
        gym.ActionWrapper.__init__(self, env)
        assert isinstance(env.action_space, gym.spaces.MultiBinary)
        self._decode_discrete_action = []
        for combo in combos:
            arr = np.array([False] * env.action_space.n)
            for button in combo:
                arr[buttons.index(button)] = True
            self._decode_discrete_action.append(arr)

        self.action_space = gym.spaces.Discrete(len(self._decode_discrete_action))

    def action(self, act):
        return self._decode_discrete_action[act].copy()


def BreakoutDiscretizer(env):
    """
    Discretize Retro Breakout-Atari2600 environment
    """
    return Discretizer(env, buttons=env.unwrapped.buttons, combos=[[None], ['BUTTON'], ['LEFT'], ['RIGHT']])


def CentipedeDiscretizer(env):
    """
    Discretize Retro Centipede-Atari2600 environment
    """
    combos = [[None], ['BUTTON'], ['UP'], ['RIGHT'],
              ['LEFT'], ['DOWN'], ['UP', 'RIGHT'], ['UP', 'LEFT'],
              ['DOWN', 'RIGHT'], ['DOWN', 'LEFT'], ['UP', 'BUTTON'], ['RIGHT', 'BUTTON'],
              ['LEFT', 'BUTTON'], ['DOWN', 'BUTTON'], ['UP', 'RIGHT', 'BUTTON'], ['UP', 'LEFT', 'BUTTON'],
              ['DOWN', 'RIGHT', 'BUTTON'], ['DOWN', 'LEFT', 'BUTTON']]
    return Discretizer(env, buttons=env.unwrapped.buttons, combos=combos)


def VideoPinballDiscretizer(env):
    """
    Discretize Retro VideoPinball-Atari2600 environment
    """
    return Discretizer(env, buttons=env.unwrapped.buttons, combos=  [[None], ['BUTTON'], ['LEFT'],
                                                                    ['RIGHT'], ['UP'], ['DOWN']])
