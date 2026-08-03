"""
Image-based (pixel) DQN on discrete LunarLander-v2.

Observation  : 4 stacked 84x84 grayscale frames rendered from the env
Action space : Discrete(4)  -> [noop, left engine, main engine, right engine]
Algorithm    : Double DQN + Dueling head + Huber loss + target network

Headless machines:
    export SDL_VIDEODRIVER=dummy      # or run under xvfb-run
Deps:
    pip install "gym[box2d]" opencv-python torch numpy
"""

import argparse
import os
import random
import time
from collections import deque

import cv2
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------------------------------------------------
# Hyperparameters
# ----------------------------------------------------------------------
ENV_ID = "LunarLander-v2"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SEED = 42
FRAME_SIZE = 84
FRAME_STACK = 4
ACTION_REPEAT = 4          # 1 agent step == 4 env steps

GAMMA = 0.99
LR = 1e-4
BUFFER_SIZE = 100_000      # in agent steps (~0.7 GB with LazyFrames)
BATCH_SIZE = 32
TOTAL_STEPS = 400_000      # agent steps  (== 1.6M env frames)
LEARN_START = 10_000       # fill buffer before any gradient step
TRAIN_EVERY = 4            # gradient step every N agent steps
TARGET_UPDATE = 2_000      # hard copy online -> target every N gradient steps
GRAD_CLIP = 10.0
REWARD_SCALE = 0.1         # only for the TD target; logs show raw returns

EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY_STEPS = 150_000  # linear anneal over agent steps
EVAL_EPS = 0.01

EVAL_EVERY = 25_000
EVAL_EPISODES = 5

# NOTE: RL has no "epoch" (there is no fixed dataset to iterate over).
# The closest unit is an *episode* -- one launch until crash / landing / timeout.
# 400k agent steps is only ~3-6k episodes, so episode-based saving is sparse;
# the step-based autosave below is what actually protects against a crash.
SAVE_EVERY_EPISODES = 200     # milestone checkpoint every N episodes (0 disables)
AUTOSAVE_EVERY_STEPS = 10_000  # overwrite qnet_latest.pt every N steps (0 disables)
KEEP_LAST_CKPTS = 0            # 0 = keep every milestone, N = keep newest N
SAVE_DIR = "dqn_lander_image_ckpt"


# ----------------------------------------------------------------------
# Gym API compatibility (works with both old <0.26 and new >=0.26 gym)
# ----------------------------------------------------------------------
def make_env(env_id, render_mode="rgb_array"):
    try:
        return gym.make(env_id, render_mode=render_mode), True
    except TypeError:
        return gym.make(env_id), False


def env_reset(env, seed=None):
    try:
        out = env.reset(seed=seed) if seed is not None else env.reset()
    except TypeError:
        if seed is not None:
            try:
                env.seed(seed)
            except Exception:
                pass
        out = env.reset()
    return out[0] if isinstance(out, tuple) else out


def env_step(env, action):
    """Returns obs, reward, terminated, truncated, info."""
    out = env.step(action)
    if len(out) == 5:
        obs, r, terminated, truncated, info = out
    else:
        obs, r, done, info = out
        truncated = bool(info.get("TimeLimit.truncated", False))
        terminated = bool(done) and not truncated
    return obs, float(r), bool(terminated), bool(truncated), info


# ----------------------------------------------------------------------
# Pixel environment wrapper
# ----------------------------------------------------------------------
class PixelLunarLander:
    """Renders the env to pixels, grayscales / resizes / stacks frames."""

    def __init__(self, env_id=ENV_ID, size=FRAME_SIZE, stack=FRAME_STACK,
                 action_repeat=ACTION_REPEAT, human=False):
        mode = "human" if human else "rgb_array"
        self.env, self.new_api = make_env(env_id, mode)
        self.size = size
        self.stack = stack
        self.action_repeat = action_repeat
        self.n_actions = self.env.action_space.n
        self.frames = deque(maxlen=stack)

    def _render_frame(self):
        rgb = self.env.render() if self.new_api else self.env.render(mode="rgb_array")
        if rgb is None:
            raise RuntimeError(
                "render() returned None. On a headless box set "
                "SDL_VIDEODRIVER=dummy or use xvfb-run."
            )
        gray = cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2GRAY)
        return cv2.resize(gray, (self.size, self.size),
                          interpolation=cv2.INTER_AREA)  # uint8 (84, 84)

    def reset(self, seed=None):
        env_reset(self.env, seed)
        f = self._render_frame()
        for _ in range(self.stack):
            self.frames.append(f)
        return LazyFrames(list(self.frames))

    def step(self, action):
        """Repeats the action; only the last frame is rendered (saves time)."""
        total_r, terminated, truncated = 0.0, False, False
        for _ in range(self.action_repeat):
            _, r, terminated, truncated, _ = env_step(self.env, action)
            total_r += r
            if terminated or truncated:
                break
        self.frames.append(self._render_frame())
        return LazyFrames(list(self.frames)), total_r, terminated, truncated

    def seed_spaces(self, seed):
        try:
            self.env.action_space.seed(seed)
        except Exception:
            pass

    def close(self):
        self.env.close()


class LazyFrames:
    """Holds references to frames so s and s' share memory in the buffer."""
    __slots__ = ("_frames",)

    def __init__(self, frames):
        self._frames = frames

    def __array__(self, dtype=None):
        out = np.stack(self._frames, axis=0)  # (stack, 84, 84) uint8
        return out.astype(dtype) if dtype is not None else out


# ----------------------------------------------------------------------
# Replay buffer
# ----------------------------------------------------------------------
class ReplayBuffer:
    def __init__(self, size):
        self.buf = deque(maxlen=size)

    def add(self, s, a, r, s2, done):
        self.buf.append((s, a, r, s2, done))

    def sample(self, batch_size):
        batch = random.sample(self.buf, batch_size)
        s, a, r, s2, d = zip(*batch)
        s = np.stack([np.asarray(x) for x in s])
        s2 = np.stack([np.asarray(x) for x in s2])
        return (s,
                np.asarray(a, dtype=np.int64),
                np.asarray(r, dtype=np.float32),
                s2,
                np.asarray(d, dtype=np.float32))

    def __len__(self):
        return len(self.buf)


# ----------------------------------------------------------------------
# Dueling CNN Q-network
# ----------------------------------------------------------------------
class QNet(nn.Module):
    def __init__(self, in_channels, n_actions):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, 8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            n_flat = self.conv(torch.zeros(1, in_channels,
                                           FRAME_SIZE, FRAME_SIZE)).shape[1]
        self.value = nn.Sequential(nn.Linear(n_flat, 512), nn.ReLU(),
                                   nn.Linear(512, 1))
        self.adv = nn.Sequential(nn.Linear(n_flat, 512), nn.ReLU(),
                                 nn.Linear(512, n_actions))

    def forward(self, x):
        # x: uint8 or float tensor (B, stack, 84, 84)
        x = x.float() / 255.0
        h = self.conv(x)
        v = self.value(h)
        a = self.adv(h)
        return v + a - a.mean(dim=1, keepdim=True)


# ----------------------------------------------------------------------
# Agent
# ----------------------------------------------------------------------
class DQNAgent:
    def __init__(self, in_channels, n_actions):
        self.n_actions = n_actions
        self.online = QNet(in_channels, n_actions).to(DEVICE)
        self.target = QNet(in_channels, n_actions).to(DEVICE)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()
        self.opt = torch.optim.Adam(self.online.parameters(), lr=LR, eps=1e-4)
        self.updates = 0

    @torch.no_grad()
    def act(self, state, eps):
        if random.random() < eps:
            return random.randrange(self.n_actions)
        s = torch.as_tensor(np.asarray(state), device=DEVICE).unsqueeze(0)
        return int(self.online(s).argmax(dim=1).item())

    def update(self, buffer):
        s, a, r, s2, d = buffer.sample(BATCH_SIZE)
        s = torch.as_tensor(s, device=DEVICE)
        s2 = torch.as_tensor(s2, device=DEVICE)
        a = torch.as_tensor(a, device=DEVICE).unsqueeze(1)
        r = torch.as_tensor(r, device=DEVICE).unsqueeze(1) * REWARD_SCALE
        d = torch.as_tensor(d, device=DEVICE).unsqueeze(1)

        q = self.online(s).gather(1, a)

        with torch.no_grad():
            # Double DQN: online picks the action, target evaluates it
            next_a = self.online(s2).argmax(dim=1, keepdim=True)
            q_next = self.target(s2).gather(1, next_a)
            backup = r + GAMMA * (1.0 - d) * q_next

        loss = F.smooth_l1_loss(q, backup)

        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), GRAD_CLIP)
        self.opt.step()

        self.updates += 1
        if self.updates % TARGET_UPDATE == 0:
            self.target.load_state_dict(self.online.state_dict())

        return float(loss.item())


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def epsilon_at(step):
    frac = min(1.0, step / EPS_DECAY_STEPS)
    return EPS_START + frac * (EPS_END - EPS_START)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_ckpt(agent, path, step, episode):
    """Full checkpoint: weights + optimizer state + counters (resumable).

    Writes to a temp file then atomically renames, so a crash mid-save
    cannot leave a corrupted checkpoint behind.
    """
    tmp = path + ".tmp"
    torch.save({
        "model": agent.online.state_dict(),
        "target": agent.target.state_dict(),
        "opt": agent.opt.state_dict(),
        "updates": agent.updates,
        "step": step,
        "episode": episode,
    }, tmp)
    os.replace(tmp, path)


def load_ckpt(agent, path):
    """Accepts both the full dict format and a bare state_dict."""
    ck = torch.load(path, map_location=DEVICE)
    if isinstance(ck, dict) and "model" in ck:
        agent.online.load_state_dict(ck["model"])
        agent.target.load_state_dict(ck.get("target", ck["model"]))
        if "opt" in ck:
            agent.opt.load_state_dict(ck["opt"])
        agent.updates = int(ck.get("updates", 0))
        return int(ck.get("step", 0)), int(ck.get("episode", 0))
    agent.online.load_state_dict(ck)
    agent.target.load_state_dict(ck)
    return 0, 0


def prune_ckpts(pattern_dir, prefix, keep):
    """Delete old periodic checkpoints, keeping only the newest `keep` files."""
    if keep <= 0:
        return
    files = sorted(
        (f for f in os.listdir(pattern_dir) if f.startswith(prefix)),
        key=lambda f: os.path.getmtime(os.path.join(pattern_dir, f)),
    )
    for f in files[:-keep]:
        try:
            os.remove(os.path.join(pattern_dir, f))
        except OSError:
            pass


@torch.no_grad()
def evaluate(agent, episodes=EVAL_EPISODES, human=False, seed=None):
    env = PixelLunarLander(human=human)
    returns = []
    for i in range(episodes):
        s = env.reset(seed=None if seed is None else seed + 1000 + i)
        ep_ret, done = 0.0, False
        while not done:
            a = agent.act(s, EVAL_EPS)
            s, r, term, trunc = env.step(a)
            ep_ret += r
            done = term or trunc
        returns.append(ep_ret)
    env.close()
    return float(np.mean(returns)), float(np.std(returns))


# ----------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------
def train(total_steps=TOTAL_STEPS, seed=SEED, resume=None,
          save_every_episodes=SAVE_EVERY_EPISODES,
          autosave_every_steps=AUTOSAVE_EVERY_STEPS,
          keep_last=KEEP_LAST_CKPTS):
    os.makedirs(SAVE_DIR, exist_ok=True)
    set_seed(seed)

    env = PixelLunarLander()
    env.seed_spaces(seed)
    agent = DQNAgent(FRAME_STACK, env.n_actions)

    start_step, episode = 0, 0
    if resume:
        start_step, episode = load_ckpt(agent, resume)
        print(f"Resumed from {resume} (step={start_step}, episode={episode}). "
              f"Note: the replay buffer is NOT saved and starts empty.")

    buffer = ReplayBuffer(BUFFER_SIZE)
    s = env.reset(seed=seed)

    ep_ret, ep_len, best_eval = 0.0, 0, -1e9
    recent = deque(maxlen=20)
    last_loss = None
    t0 = time.time()

    for step in range(start_step + 1, total_steps + 1):
        eps = 1.0 if step < LEARN_START else epsilon_at(step - LEARN_START)
        a = agent.act(s, eps)
        s2, r, terminated, truncated = env.step(a)
        ep_ret += r
        ep_len += 1

        # Bootstrap through time-limit truncation, cut only on real termination
        buffer.add(s, a, r, s2, float(terminated))
        s = s2

        if terminated or truncated:
            episode += 1
            recent.append(ep_ret)

            # ---- milestone checkpoint, one file per N episodes ----
            if save_every_episodes > 0 and episode % save_every_episodes == 0:
                path = os.path.join(SAVE_DIR, f"qnet_ep{episode:06d}.pt")
                save_ckpt(agent, path, step, episode)
                print(f"  [milestone] episode {episode} (step {step}) -> "
                      f"{os.path.basename(path)} | "
                      f"Last20Avg={np.mean(recent):.1f}")
                prune_ckpts(SAVE_DIR, "qnet_ep", keep_last)

            s = env.reset()
            ep_ret, ep_len = 0.0, 0

        if step >= LEARN_START and step % TRAIN_EVERY == 0:
            last_loss = agent.update(buffer)

        if step % 5_000 == 0:
            fps = step * ACTION_REPEAT / (time.time() - t0)
            avg20 = np.mean(recent) if recent else float("nan")
            loss_str = f"{last_loss:.4f}" if last_loss is not None else "N/A"
            print(f"[{step}/{total_steps}] ep={episode} | eps={eps:.3f} | "
                  f"Last20Avg={avg20:7.1f} | loss={loss_str} | "
                  f"buffer={len(buffer)} | env_fps={fps:.0f}")

        # ---- rolling autosave (crash recovery): always one small file ----
        if autosave_every_steps > 0 and step % autosave_every_steps == 0:
            save_ckpt(agent, os.path.join(SAVE_DIR, "qnet_latest.pt"),
                      step, episode)

        if step % EVAL_EVERY == 0 and step >= LEARN_START:
            m, sd = evaluate(agent, seed=seed)
            print(f"  >> eval @ step {step} (ep {episode}): {m:.1f} +/- {sd:.1f}")
            if m > best_eval:
                best_eval = m
                save_ckpt(agent, os.path.join(SAVE_DIR, "qnet_best.pt"),
                          step, episode)
                print(f"  >> new best ({m:.1f}), saved qnet_best.pt")

    env.close()
    save_ckpt(agent, os.path.join(SAVE_DIR, "qnet_final.pt"), total_steps, episode)
    print(f"Training finished. {episode} episodes over {total_steps} agent steps "
          f"({total_steps * ACTION_REPEAT} env frames).")


def play(ckpt, episodes=5, human=True):
    env = PixelLunarLander()
    agent = DQNAgent(FRAME_STACK, env.n_actions)
    env.close()
    step, episode = load_ckpt(agent, ckpt)
    agent.online.eval()
    print(f"loaded {ckpt} (step={step}, episode={episode})")
    m, sd = evaluate(agent, episodes=episodes, human=human)
    print(f"{episodes} episodes: {m:.1f} +/- {sd:.1f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["train", "play"], default="train")
    p.add_argument("--steps", type=int, default=TOTAL_STEPS)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--ckpt", type=str, default=None)
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--save-every-episodes", type=int, default=SAVE_EVERY_EPISODES,
                   help="milestone checkpoint every N episodes (0 disables)")
    p.add_argument("--autosave-every-steps", type=int, default=AUTOSAVE_EVERY_STEPS,
                   help="overwrite qnet_latest.pt every N agent steps (0 disables)")
    p.add_argument("--keep-last", type=int, default=KEEP_LAST_CKPTS,
                   help="keep only the newest N milestone checkpoints (0 = all)")
    p.add_argument("--no-window", action="store_true",
                   help="play without opening a render window")
    args = p.parse_args()

    print(f"device: {DEVICE}")
    if args.mode == "train":
        train(total_steps=args.steps, seed=args.seed, resume=args.ckpt,
              save_every_episodes=args.save_every_episodes,
              autosave_every_steps=args.autosave_every_steps,
              keep_last=args.keep_last)
    else:
        ckpt = args.ckpt or os.path.join(SAVE_DIR, "qnet_best.pt")
        play(ckpt, episodes=args.episodes, human=not args.no_window)
