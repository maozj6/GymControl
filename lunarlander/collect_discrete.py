"""
Collect an image-based trajectory dataset from discrete LunarLander-v2.

Default plan: 100,000 trajectories
    - 20,000 from a uniform random policy
    - 80,000 from 8 parameterised heuristic controllers (10,000 each)

Each trajectory is written to its own .npz containing images, actions, states,
rewards, terrain geometry and the seed needed to replay it exactly.

Headless machines:
    export SDL_VIDEODRIVER=dummy      # or run under xvfb-run
Deps:
    pip install "gym[box2d]" opencv-python numpy

Usage:
    python collect_lander_data.py --estimate            # measure size/time first
    python collect_lander_data.py --workers 8
    python collect_lander_data.py --inspect data/shard_000/traj_0000000.npz
"""

import argparse
import json
import multiprocessing as mp
import os
import random
import time

import cv2
import gym
import numpy as np

# ----------------------------------------------------------------------
# Defaults
# ----------------------------------------------------------------------
ENV_ID = "LunarLander-v2"
OUT_DIR = "lander_dataset"
N_RANDOM = 20_000
N_PER_CONTROLLER = 10_000     # x 8 controllers = 80,000
FRAME_SIZE = 84
GRAYSCALE = True
ACTION_REPEAT = 4             # 1 stored step == 4 env steps (50Hz -> 12.5Hz)
MAX_STORED_STEPS = 300        # hard cap per trajectory
SHARD_SIZE = 1_000            # files per subdirectory
COMPRESS = True
BASE_SEED = 100_000

# 8-dim observation layout, kept here so the npz is self-describing
STATE_KEYS = ["x", "y", "vx", "vy", "angle", "angular_vel",
              "leg1_contact", "leg2_contact"]
ACTION_MEANINGS = ["noop", "left_engine", "main_engine", "right_engine"]


# ----------------------------------------------------------------------
# 8 controllers: same PD structure, different gains + action noise.
# The point is behavioural diversity, not 8 copies of one expert.
# ----------------------------------------------------------------------
BASE_PARAMS = dict(ang_x=0.5, ang_vx=1.0, ang_clip=0.4, hover_k=0.55,
                   ang_p=0.5, ang_d=1.0, hov_p=0.5, hov_d=0.5,
                   main_thresh=0.05, side_thresh=0.05, eps=0.0)


def _p(**over):
    p = dict(BASE_PARAMS)
    p.update(over)
    return p


CONTROLLERS = {
    # near-optimal reference
    "expert":        _p(),
    # same policy, mild exploration -> covers off-expert states
    "expert_noisy":  _p(eps=0.05),
    # twitchy: high gains, overshoots and corrects hard
    "aggressive":    _p(ang_p=1.0, ang_d=1.6, hov_p=0.9, hov_d=0.8,
                        main_thresh=0.02, side_thresh=0.02),
    # slow to react, tends to drift and land off-pad
    "sluggish":      _p(ang_p=0.25, ang_d=0.5, hov_p=0.25, hov_d=0.3,
                        main_thresh=0.12, side_thresh=0.12),
    # burns fuel holding altitude, long episodes, often times out
    "hoverer":       _p(hover_k=1.1, hov_p=0.8, hov_d=0.9, main_thresh=0.01),
    # descends fast, frequently hard-lands or crashes
    "fast_descent":  _p(hover_k=0.25, hov_p=0.35, hov_d=0.25,
                        main_thresh=0.15),
    # systematic tilt bias -> approaches the pad from one side
    "tilted":        _p(ang_x=0.85, ang_clip=0.6, ang_bias=0.12),
    # heavy noise: closest thing to a "bad demonstrator"
    "very_noisy":    _p(eps=0.25),
}
CONTROLLER_IDS = {name: i for i, name in enumerate(sorted(CONTROLLERS))}


def heuristic_action(s, p, rng):
    """Parameterised PD controller over the 8-dim LunarLander observation."""
    if p["eps"] > 0.0 and rng.random() < p["eps"]:
        return rng.randrange(4)

    angle_targ = s[0] * p["ang_x"] + s[2] * p["ang_vx"] + p.get("ang_bias", 0.0)
    angle_targ = float(np.clip(angle_targ, -p["ang_clip"], p["ang_clip"]))
    hover_targ = p["hover_k"] * abs(s[0])

    angle_todo = (angle_targ - s[4]) * p["ang_p"] - s[5] * p["ang_d"]
    hover_todo = (hover_targ - s[1]) * p["hov_p"] - s[3] * p["hov_d"]

    if s[6] or s[7]:                       # a leg is touching down
        angle_todo = 0.0
        hover_todo = -s[3] * 0.5

    if hover_todo > abs(angle_todo) and hover_todo > p["main_thresh"]:
        return 2
    if angle_todo < -p["side_thresh"]:
        return 3
    if angle_todo > p["side_thresh"]:
        return 1
    return 0


# ----------------------------------------------------------------------
# Gym API compatibility (old <0.26 and new >=0.26)
# ----------------------------------------------------------------------
def make_env(env_id=ENV_ID):
    try:
        return gym.make(env_id, render_mode="rgb_array"), True
    except TypeError:
        return gym.make(env_id), False


def env_reset(env, seed):
    try:
        out = env.reset(seed=seed)
    except TypeError:
        try:
            env.seed(seed)
        except Exception:
            pass
        out = env.reset()
    return out[0] if isinstance(out, tuple) else out


def env_step(env, action):
    out = env.step(action)
    if len(out) == 5:
        obs, r, terminated, truncated, info = out
    else:
        obs, r, done, info = out
        truncated = bool(info.get("TimeLimit.truncated", False))
        terminated = bool(done) and not truncated
    return obs, float(r), bool(terminated), bool(truncated)


# ----------------------------------------------------------------------
# Frame + terrain extraction
# ----------------------------------------------------------------------
def render_frame(env, new_api, size=FRAME_SIZE, gray=GRAYSCALE):
    rgb = env.render() if new_api else env.render(mode="rgb_array")
    if rgb is None:
        raise RuntimeError("render() returned None -- set SDL_VIDEODRIVER=dummy "
                           "or run under xvfb-run.")
    img = np.asarray(rgb)
    if gray:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)


def extract_terrain(env):
    """Ground polyline + helipad location. Must be called after reset().

    LunarLander builds the moon from CHUNKS-1 edge fixtures; sky_polys[i]
    is [p1, p2, (p2.x, H), (p1.x, H)], so p1/p2 give the ground vertices.
    """
    u = env.unwrapped
    out = {}
    sky = getattr(u, "sky_polys", None)
    if sky:
        xs = [float(poly[0][0]) for poly in sky] + [float(sky[-1][1][0])]
        ys = [float(poly[0][1]) for poly in sky] + [float(sky[-1][1][1])]
        out["terrain_x"] = np.asarray(xs, dtype=np.float32)
        out["terrain_y"] = np.asarray(ys, dtype=np.float32)
    for k in ("helipad_x1", "helipad_x2", "helipad_y"):
        v = getattr(u, k, None)
        if v is not None:
            out[k] = np.float32(v)
    return out


def lander_raw_state(env):
    """True world-frame pose, un-normalised (obs is scaled/centred)."""
    lander = getattr(env.unwrapped, "lander", None)
    if lander is None:
        return np.zeros(6, dtype=np.float32)
    return np.asarray([lander.position.x, lander.position.y,
                       lander.linearVelocity.x, lander.linearVelocity.y,
                       lander.angle, lander.angularVelocity], dtype=np.float32)


# ----------------------------------------------------------------------
# Rollout
# ----------------------------------------------------------------------
def rollout(env, new_api, seed, policy, cfg):
    """Run one episode. `policy` is a controller name or 'random'."""
    rng = random.Random(seed)
    params = None if policy == "random" else CONTROLLERS[policy]

    s = env_reset(env, seed)
    terrain = extract_terrain(env)

    images = [render_frame(env, new_api, cfg["size"], cfg["gray"])]
    states = [np.asarray(s, dtype=np.float32)]
    raws = [lander_raw_state(env)]
    actions, rewards = [], []
    terminated = truncated = False

    for _ in range(cfg["max_steps"]):
        a = rng.randrange(4) if params is None else heuristic_action(s, params, rng)

        r_sum = 0.0
        for _ in range(cfg["repeat"]):
            s, r, terminated, truncated = env_step(env, a)
            r_sum += r
            if terminated or truncated:
                break

        actions.append(a)
        rewards.append(r_sum)
        states.append(np.asarray(s, dtype=np.float32))
        raws.append(lander_raw_state(env))
        images.append(render_frame(env, new_api, cfg["size"], cfg["gray"]))

        if terminated or truncated:
            break
    else:
        truncated = True   # hit our own MAX_STORED_STEPS cap

    data = {
        # T+1 frames / states so the final action also has a successor
        "image": np.asarray(images, dtype=np.uint8),
        "state": np.asarray(states, dtype=np.float32),
        "lander_raw": np.asarray(raws, dtype=np.float32),
        "action": np.asarray(actions, dtype=np.int64),
        "reward": np.asarray(rewards, dtype=np.float32),
        "terminated": np.bool_(terminated),
        "truncated": np.bool_(truncated),
        "seed": np.int64(seed),
        "policy": np.str_(policy),
        "controller_id": np.int64(-1 if policy == "random"
                                  else CONTROLLER_IDS[policy]),
        "episode_return": np.float32(float(np.sum(rewards))),
        "episode_length": np.int64(len(actions)),
        "action_repeat": np.int64(cfg["repeat"]),
        "frame_size": np.int64(cfg["size"]),
        "state_keys": np.asarray(STATE_KEYS),
        "action_meanings": np.asarray(ACTION_MEANINGS),
        "env_id": np.str_(ENV_ID),
    }
    data.update(terrain)
    if params is not None:
        data["controller_params"] = np.str_(json.dumps(params, sort_keys=True))
    return data


# ----------------------------------------------------------------------
# Worker plumbing
# ----------------------------------------------------------------------
_ENV = None
_NEW_API = False


def _init_worker():
    global _ENV, _NEW_API
    _ENV, _NEW_API = make_env()


def _run_one(job):
    idx, policy, seed, cfg = job
    path = traj_path(cfg["out_dir"], idx)
    if cfg["resume"] and os.path.exists(path):
        return {"idx": idx, "skipped": True}

    try:
        data = rollout(_ENV, _NEW_API, seed, policy, cfg)
    except Exception as e:                       # keep the run alive
        return {"idx": idx, "error": f"{type(e).__name__}: {e}"}

    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp.npz"
    saver = np.savez_compressed if cfg["compress"] else np.savez
    saver(tmp, **data)
    os.replace(tmp, path)                        # atomic: no half-written npz

    return {
        "idx": idx, "path": os.path.relpath(path, cfg["out_dir"]),
        "policy": policy, "seed": seed,
        "length": int(data["episode_length"]),
        "return": float(data["episode_return"]),
        "terminated": bool(data["terminated"]),
        "bytes": os.path.getsize(path),
    }


def traj_path(out_dir, idx):
    return os.path.join(out_dir, f"shard_{idx // SHARD_SIZE:04d}",
                        f"traj_{idx:07d}.npz")


def build_jobs(cfg, n_random, n_per_ctrl):
    """Interleave policies so any prefix of the dataset is already balanced."""
    jobs = [("random", i) for i in range(n_random)]
    for name in sorted(CONTROLLERS):
        jobs += [(name, i) for i in range(n_per_ctrl)]
    rng = random.Random(0)
    rng.shuffle(jobs)
    return [(i, pol, BASE_SEED + i, cfg) for i, (pol, _) in enumerate(jobs)]


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def collect(cfg, n_random, n_per_ctrl, workers):
    os.makedirs(cfg["out_dir"], exist_ok=True)
    jobs = build_jobs(cfg, n_random, n_per_ctrl)
    total = len(jobs)
    manifest_path = os.path.join(cfg["out_dir"], "manifest.jsonl")

    print(f"planning {total} trajectories "
          f"({n_random} random + {n_per_ctrl} x {len(CONTROLLERS)} controllers)")
    print(f"output: {cfg['out_dir']}  workers: {workers}")

    t0 = time.time()
    done = skipped = failed = 0
    total_bytes = total_len = 0

    ctx = mp.get_context("spawn")
    with ctx.Pool(workers, initializer=_init_worker) as pool, \
            open(manifest_path, "a") as mf:
        for res in pool.imap_unordered(_run_one, jobs, chunksize=8):
            done += 1
            if res.get("skipped"):
                skipped += 1
            elif "error" in res:
                failed += 1
                if failed <= 10:
                    print(f"  [warn] traj {res['idx']}: {res['error']}")
            else:
                mf.write(json.dumps(res) + "\n")
                total_bytes += res["bytes"]
                total_len += res["length"]

            if done % 500 == 0 or done == total:
                el = time.time() - t0
                rate = done / max(el, 1e-9)
                eta = (total - done) / max(rate, 1e-9)
                written = done - skipped - failed
                gb = total_bytes / 1e9
                proj = gb / max(written, 1) * total
                print(f"[{done}/{total}] {rate:.1f} traj/s | "
                      f"elapsed {el/60:.1f}m | eta {eta/60:.1f}m | "
                      f"{gb:.2f}GB so far (~{proj:.1f}GB total) | "
                      f"avg_len={total_len/max(written,1):.0f} | "
                      f"skipped={skipped} failed={failed}")

            mf.flush()

    meta = {
        "env_id": ENV_ID, "n_total": total, "n_random": n_random,
        "n_per_controller": n_per_ctrl, "controllers": CONTROLLERS,
        "controller_ids": CONTROLLER_IDS, "state_keys": STATE_KEYS,
        "action_meanings": ACTION_MEANINGS, "base_seed": BASE_SEED,
        "shard_size": SHARD_SIZE, **cfg,
    }
    with open(os.path.join(cfg["out_dir"], "dataset_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\ndone in {(time.time()-t0)/60:.1f} min | "
          f"written={done-skipped-failed} skipped={skipped} failed={failed} | "
          f"{total_bytes/1e9:.2f} GB")


def estimate(cfg, n=60, workers=4):
    """Run a small sample and extrapolate size + wall-clock to the full run."""
    print(f"sampling {n} trajectories to estimate the full run...\n")
    sample_cfg = dict(cfg)
    sample_cfg["out_dir"] = os.path.join(cfg["out_dir"], "_estimate")
    os.makedirs(sample_cfg["out_dir"], exist_ok=True)

    names = ["random"] + sorted(CONTROLLERS)
    jobs = [(i, names[i % len(names)], BASE_SEED + 10**7 + i, sample_cfg)
            for i in range(n)]

    t0 = time.time()
    ctx = mp.get_context("spawn")
    with ctx.Pool(workers, initializer=_init_worker) as pool:
        res = [r for r in pool.map(_run_one, jobs) if "error" not in r]
    el = time.time() - t0

    if not res:
        print("all sample rollouts failed -- check the render setup.")
        return

    per_traj_bytes = np.mean([r["bytes"] for r in res])
    per_traj_sec = el / len(res) * workers          # single-worker cost
    total = N_RANDOM + N_PER_CONTROLLER * len(CONTROLLERS)

    print(f"avg length : {np.mean([r['length'] for r in res]):.0f} stored steps")
    print(f"avg size   : {per_traj_bytes/1024:.1f} KB / trajectory")
    print(f"avg cost   : {per_traj_sec*1000:.0f} ms / trajectory (1 worker)\n")
    print(f"--> {total} trajectories ~= {per_traj_bytes*total/1e9:.1f} GB")
    for w in (1, 4, 8, 16):
        print(f"--> with {w:2d} workers: {per_traj_sec*total/w/3600:.1f} hours")
    print(f"\nsample files left in {sample_cfg['out_dir']} (safe to delete)")


def inspect(path):
    d = np.load(path, allow_pickle=False)
    print(f"{path}\n")
    for k in d.files:
        v = d[k]
        if v.ndim == 0:
            print(f"  {k:20s} scalar {v.dtype}  = {v}")
        else:
            print(f"  {k:20s} {str(v.shape):18s} {v.dtype}")
    print(f"\n  file size: {os.path.getsize(path)/1024:.1f} KB")
    print(f"  return={float(d['episode_return']):.1f}  "
          f"len={int(d['episode_length'])}  "
          f"terminated={bool(d['terminated'])}  "
          f"truncated={bool(d['truncated'])}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default=OUT_DIR)
    p.add_argument("--n-random", type=int, default=N_RANDOM)
    p.add_argument("--n-per-controller", type=int, default=N_PER_CONTROLLER)
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    p.add_argument("--size", type=int, default=FRAME_SIZE)
    p.add_argument("--rgb", action="store_true", help="store RGB instead of grayscale")
    p.add_argument("--action-repeat", type=int, default=ACTION_REPEAT)
    p.add_argument("--max-steps", type=int, default=MAX_STORED_STEPS)
    p.add_argument("--no-compress", action="store_true")
    p.add_argument("--no-resume", action="store_true",
                   help="re-generate trajectories even if the file exists")
    p.add_argument("--estimate", action="store_true",
                   help="sample a few episodes and project size / runtime")
    p.add_argument("--inspect", type=str, default=None)
    args = p.parse_args()

    if args.inspect:
        inspect(args.inspect)
        raise SystemExit

    cfg = {
        "out_dir": args.out_dir,
        "size": args.size,
        "gray": not args.rgb,
        "repeat": args.action_repeat,
        "max_steps": args.max_steps,
        "compress": not args.no_compress,
        "resume": not args.no_resume,
    }

    if args.estimate:
        estimate(cfg, workers=min(4, args.workers))
    else:
        collect(cfg, args.n_random, args.n_per_controller, args.workers)
