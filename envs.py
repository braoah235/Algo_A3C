from __future__ import annotations

import os
from collections import deque
from typing import Any, Tuple

import numpy as np

try:
    import gymnasium as gym
except ImportError:  # pragma: no cover
    import gym  # type: ignore

try:
    import ale_py
except ImportError:  # pragma: no cover
    ale_py = None

try:
    from PIL import Image
except ImportError as exc:  # pragma: no cover
    raise ImportError("Pillow is required for Atari preprocessing. Install it with: pip install pillow") from exc


class GymCompatWrapper(gym.Wrapper):
    """Expose old Gym API expected by this legacy A3C codebase."""

    def seed(self, seed: int | None = None):
        try:
            self.reset(seed=seed)
        except TypeError:
            if hasattr(self.env, "seed"):
                self.env.seed(seed)
        return [seed]

    def reset(self, **kwargs):
        out = self.env.reset(**kwargs)
        if isinstance(out, tuple) and len(out) == 2:
            obs, _info = out
            return obs
        return out

    def step(self, action):
        if isinstance(action, np.ndarray):
            action = int(action.item())
        out = self.env.step(action)
        if isinstance(out, tuple) and len(out) == 5:
            obs, reward, terminated, truncated, info = out
            done = terminated or truncated
            return obs, reward, done, info
        return out


class NoopResetEnv(gym.Wrapper):
    """Sample initial states by taking random number of no-op actions on reset."""

    def __init__(self, env, noop_max: int = 30):
        super().__init__(env)
        self.noop_max = max(1, int(noop_max))
        self.noop_action = 0

    def reset(self, **kwargs):
        out = self.env.reset(**kwargs)
        if isinstance(out, tuple) and len(out) == 2:
            obs, info = out
        else:
            obs, info = out, {}

        noops = np.random.randint(1, self.noop_max + 1)
        for _ in range(noops):
            step_out = self.env.step(self.noop_action)
            if isinstance(step_out, tuple) and len(step_out) == 5:
                obs, _reward, terminated, truncated, info = step_out
                done = terminated or truncated
            else:
                obs, _reward, done, info = step_out
            if done:
                out = self.env.reset(**kwargs)
                if isinstance(out, tuple) and len(out) == 2:
                    obs, info = out
                else:
                    obs, info = out, {}

        if isinstance(out, tuple) and len(out) == 2:
            return obs, info
        return obs


class FireResetEnv(gym.Wrapper):
    """Take FIRE actions on reset for environments that require it (e.g. Breakout)."""

    def reset(self, **kwargs):
        out = self.env.reset(**kwargs)
        if isinstance(out, tuple) and len(out) == 2:
            obs, info = out
        else:
            obs, info = out, {}

        for action in (1, 2):
            step_out = self.env.step(action)
            if isinstance(step_out, tuple) and len(step_out) == 5:
                obs, _reward, terminated, truncated, info = step_out
                done = terminated or truncated
            else:
                obs, _reward, done, info = step_out
            if done:
                out = self.env.reset(**kwargs)
                if isinstance(out, tuple) and len(out) == 2:
                    obs, info = out
                else:
                    obs, info = out, {}

        if isinstance(out, tuple) and len(out) == 2:
            return obs, info
        return obs


class AtariRescale42x42(gym.ObservationWrapper):
    """Convert Atari RGB frames to normalized grayscale 42x42, channel-first."""

    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(1, 42, 42),
            dtype=np.float32,
        )

    def observation(self, observation):
        # Convert RGB to grayscale.
        gray = np.dot(observation[..., :3], [0.299, 0.587, 0.114])
        frame = Image.fromarray(gray.astype(np.uint8))
        # Same style as original A3C Atari preprocessing.
        frame = frame.resize((80, 80), Image.BILINEAR)
        frame = frame.crop((18, 18, 60, 60))
        frame = frame.resize((42, 42), Image.BILINEAR)
        arr = np.asarray(frame, dtype=np.float32) / 255.0
        return arr[np.newaxis, :, :]


class NormalizedEnv(gym.ObservationWrapper):
    """Online observation normalization used in classic A3C implementations."""

    def __init__(self, env, alpha: float = 0.9999):
        super().__init__(env)
        self.alpha = alpha
        self.state_mean = 0.0
        self.state_std = 0.0
        self.num_steps = 0

    def observation(self, observation):
        self.num_steps += 1
        self.state_mean = self.state_mean * self.alpha + observation.mean() * (1 - self.alpha)
        self.state_std = self.state_std * self.alpha + observation.std() * (1 - self.alpha)

        unbiased_mean = self.state_mean / (1 - pow(self.alpha, self.num_steps))
        unbiased_std = self.state_std / (1 - pow(self.alpha, self.num_steps))

        return (observation - unbiased_mean) / (unbiased_std + 1e-8)


class FrameStackChannels(gym.Wrapper):
    """Stack the last K observations along the channel axis (C,H,W)."""

    def __init__(self, env, k: int = 4):
        super().__init__(env)
        self.k = max(1, int(k))
        self.frames = deque(maxlen=self.k)
        c, h, w = self.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(c * self.k, h, w),
            dtype=np.float32,
        )

    def reset(self, **kwargs):
        out = self.env.reset(**kwargs)
        if isinstance(out, tuple) and len(out) == 2:
            obs, info = out
        else:
            obs, info = out, {}
        for _ in range(self.k):
            self.frames.append(obs)
        stacked = self._get_stacked_obs()
        if isinstance(out, tuple) and len(out) == 2:
            return stacked, info
        return stacked

    def step(self, action):
        out = self.env.step(action)
        if isinstance(out, tuple) and len(out) == 5:
            obs, reward, terminated, truncated, info = out
            self.frames.append(obs)
            return self._get_stacked_obs(), reward, terminated, truncated, info
        obs, reward, done, info = out
        self.frames.append(obs)
        return self._get_stacked_obs(), reward, done, info

    def _get_stacked_obs(self):
        return np.concatenate(list(self.frames), axis=0).astype(np.float32, copy=False)


def _make_atari_env(env_name: str, video: bool = False):
    # Gymnasium 1.x may require explicit ALE registration.
    if ale_py is not None and hasattr(gym, "register_envs"):
        try:
            gym.register_envs(ale_py)
        except Exception:
            pass

    candidates = [env_name]
    if env_name == "Breakout-v0":
        candidates = [
            "ALE/Breakout-v5",
            "BreakoutNoFrameskip-v4",
            "Breakout-v4",
            env_name,
        ]

    last_error: Exception | None = None
    for candidate in candidates:
        try:
            if video:
                # Gymnasium RecordVideo expects rgb_array rendering.
                return gym.make(candidate, render_mode="rgb_array")
            return gym.make(candidate)
        except Exception as exc:  # pragma: no cover
            last_error = exc

    raise RuntimeError(
        f"Unable to create Atari env from candidates={candidates}. "
        "Install Atari deps first, e.g. pip install gymnasium[atari,accept-rom-license] ale-py"
    ) from last_error


def create_atari_env(
    env_name: str,
    video: bool = False,
    stack_frames: int = 1,
):
    env = _make_atari_env(env_name, video=video)

    # Classic Atari reset handling used by many A3C baselines.
    env = NoopResetEnv(env, noop_max=30)
    try:
        action_meanings = env.unwrapped.get_action_meanings()
    except Exception:
        action_meanings = []
    if "FIRE" in action_meanings:
        env = FireResetEnv(env)

    if video:
        os.makedirs("test", exist_ok=True)
        try:
            env = gym.wrappers.RecordVideo(
                env,
                video_folder="test",
                episode_trigger=lambda episode_id: episode_id % 25 == 0,
            )
        except Exception:
            # Older gym versions may not provide RecordVideo.
            pass

    env = AtariRescale42x42(env)
    env = NormalizedEnv(env)
    env = FrameStackChannels(env, k=stack_frames)
    env = GymCompatWrapper(env)
    return env
