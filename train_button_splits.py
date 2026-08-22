from __future__ import annotations
import csv
import json
import os
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from types import SimpleNamespace
import gymnasium as gym
import metaworld # type: ignore
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecNormalize


# =============================================================================
# Defaults
# =============================================================================

ENV_ID = "Meta-World/MT1"
ENV_NAME = "button-press-v3"
REWARD_TYPE = "v2"
MAX_EPISODE_STEPS = 500
TASK_SEED = 67
START_SPLIT_SEED = 67
N_SPLITS = 5
TRAIN_TEST_SPLIT = (45, 5)
N_ENVS = 4
TOTAL_TIMESTEPS = 500_000
CHECKPOINT_FREQ = 5_000
TRAIN_SEEDS = [11]

OUTPUT_DIR = "./button-press_v3_ppo_split_runs"

PPO_CONFIGS: Dict[str, Dict[str, Any]] = {
    "config_A": {
        "learning_rate": 3e-4,
        "n_steps": 1024,          # rollout = 4096 with 4 envs
        "batch_size": 256,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.20,
        "ent_coef": 0.0,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
    },
    "config_B": {
        "learning_rate": 1e-4,
        "n_steps": 1024,          # rollout = 4096 with 4 envs
        "batch_size": 512,
        "n_epochs": 15,
        "gamma": 0.995,
        "gae_lambda": 0.95,
        "clip_range": 0.15,
        "ent_coef": 0.0,
        "vf_coef": 0.7,
        "max_grad_norm": 0.5,
    },
    "config_C": {
        "learning_rate": 3e-4,
        "n_steps": 512,           # rollout = 2048 with 4 envs
        "batch_size": 256,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.20,
        "ent_coef": 0.0,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
    },
    "config_D": {
        "learning_rate": 2.5e-4,
        "n_steps": 1024,          # rollout = 4096 with 4 envs
        "batch_size": 256,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.20,
        "ent_coef": 0.002,        # light exploration
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
    },
    "config_E": {  # paper-like / higher-learning-rate PPO
        "learning_rate": 5e-4,
        "n_steps": 1024,
        "batch_size": 256,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.20,
        "ent_coef": 0.0,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
    },
}

RUN_TOTAL_TIMESTEPS = TOTAL_TIMESTEPS
RUN_N_ENVS = N_ENVS
RUN_N_SPLITS = N_SPLITS

RUN_TASK_SEED = TASK_SEED
RUN_START_SPLIT_SEED = START_SPLIT_SEED

RUN_TRAIN_SEEDS = TRAIN_SEEDS

RUN_CONFIGS =  ["config_A","config_B","config_C","config_D","config_E"]

RUN_OUTPUT_DIR = OUTPUT_DIR
RUN_CHECKPOINT_FREQ = CHECKPOINT_FREQ

RUN_REWARD_TYPE = REWARD_TYPE
RUN_MAX_EPISODE_STEPS = MAX_EPISODE_STEPS

RUN_DEVICE = "cpu"

RUN_START_METHOD = None


RUN_USE_VECNORMALIZE = True
RUN_TRAIN_TERMINATE_ON_SUCCESS = False
RUN_EVAL_TERMINATE_ON_SUCCESS = True

# Keep progress bar and tracking.
RUN_SHOW_PROGRESS_BAR = True

#If False, existing final models are skipped.
#If True, existing final models are retrained and overwritten.
RUN_OVERWRITE_EXISTING = False

# Evaluation after each trained model.
RUN_EVAL_TRAIN_EPISODES_PER_TASK = 1
RUN_EVAL_TEST_EPISODES_PER_TASK = 1
RUN_EVAL_SEED = 100

#convert coma separated string to list of integers
def parse_int_list(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]

#convert coma separated string into list of strings
def parse_str_list(s: str) -> List[str]:
    return [x.strip() for x in s.split(",") if x.strip()]

#create dir if it doesnt exist
def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p

#add row to csv file
def append_csv_row(csv_path: Path, row: Dict[str, Any]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)

#convert values into json.Used for saving train/test split values (goal positions) into a friendly format
def to_jsonable(x: Any) -> Any:
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.integer, np.floating)):
        return x.item()
    if isinstance(x, (list, tuple)):
        return [to_jsonable(v) for v in x]
    if isinstance(x, dict):
        return {str(k): to_jsonable(v) for k, v in x.items()}
    try:
        json.dumps(x)
        return x
    except TypeError:
        return str(x)

#find the internal metaworld wrapper that has the task (goall) variations
def get_task_wrapper(env: gym.Env):
    ## Find the Meta-World wrapper that stores tasks and task sampling controls.
    cur = env
    seen = set()
    while True:
        if id(cur) in seen:
            break
        seen.add(id(cur))

        if hasattr(cur, "tasks") and hasattr(cur, "toggle_sample_tasks_on_reset"):
            return cur

        if hasattr(cur, "env"):
            cur = cur.env
            continue

        unwrapped = getattr(cur, "unwrapped", None)
        if unwrapped is not None and unwrapped is not cur:
            cur = unwrapped
            continue

        break

    raise RuntimeError("Could not find Meta-World task wrapper.")

## function for mt1 env init
def make_mt1_env(seed: int,tasks: Optional[Sequence[Any]] = None,sample_on_reset: bool = True,terminate_on_success: bool = False,render_mode: Optional[str] = None,reward_type: str = REWARD_TYPE,max_episode_steps: int = MAX_EPISODE_STEPS
) -> gym.Env:
    env = gym.make(
        ENV_ID,
        env_name=ENV_NAME,
        task_select="pseudorandom",
        terminate_on_success=terminate_on_success,
        max_episode_steps=max_episode_steps,
        seed=seed,
        render_mode=render_mode,
        reward_function_version=reward_type,
    )

    if tasks is not None:
        wrapper = get_task_wrapper(env)
        wrapper.tasks = list(tasks)
        wrapper.toggle_sample_tasks_on_reset(sample_on_reset)
    else:
        try:
            env.get_wrapper_attr("toggle_sample_tasks_on_reset")(sample_on_reset)
        except Exception:
            pass

    return env

## used for SubProcVecEnv
def make_train_env_fn( rank: int, base_seed: int, tasks: Sequence[Any], terminate_on_success: bool, reward_type: str, max_episode_steps: int,):
    def _init() -> gym.Env:
        env = make_mt1_env(
            seed=base_seed + rank,
            tasks=tasks,
            sample_on_reset=True,
            terminate_on_success=terminate_on_success,
            render_mode=None,
            reward_type=reward_type,
            max_episode_steps=max_episode_steps,
        )
        return Monitor(env)

    return _init


def extract_all_tasks(task_seed: int) -> List[Any]:
    env = make_mt1_env(seed=task_seed, tasks=None, sample_on_reset=True)
    wrapper = get_task_wrapper(env)
    tasks = list(wrapper.tasks)
    env.close()

    if len(tasks) != 50:
        raise RuntimeError(f"Expected 50 MT1 tasks, got {len(tasks)}")

    return tasks

# extracts task info 
def decode_task_data(task: Any) -> Dict[str, Any]:
    """Decode Meta-World task.data when possible, for split manifest/debugging."""
    out: Dict[str, Any] = {}

    if hasattr(task, "env_name"):
        out["task_env_name"] = str(task.env_name)

    raw = getattr(task, "data", None)
    if raw is None:
        out["decoded_data"] = None
        return out

    try:
        data = pickle.loads(raw)
        out["decoded_data"] = to_jsonable(data)

        if isinstance(data, dict):
            # Common useful fields. These may vary by Meta-World version.
            for key in [
                "rand_vec",
                "_target_pos",
                "target_pos",
                "goal",
                "obj_init_pos",
                "_obj_init_pos",
                "hand_init_pos",
            ]:
                if key in data:
                    out[key] = to_jsonable(data[key])
    except Exception as exc:
        out["decoded_data"] = f"Could not decode task.data: {exc}"

    return out


def describe_task_short(task: Any) -> str:
    d = decode_task_data(task)
    for key in ["rand_vec", "_target_pos", "target_pos", "goal", "obj_init_pos", "_obj_init_pos"]:
        if key in d:
            return f"{key}={d[key]}"
    return str(d.get("decoded_data", "<no decoded task data>"))[:160]


#create train/test split
def build_split(tasks: Sequence[Any], split_seed: int) -> Tuple[np.ndarray, np.ndarray, List[Any], List[Any]]:
    rng = np.random.default_rng(split_seed)
    indices = np.arange(len(tasks))
    rng.shuffle(indices)

    train_count, test_count = TRAIN_TEST_SPLIT
    train_idx = indices[:train_count]
    test_idx = indices[train_count : train_count + test_count]

    return train_idx, test_idx, [tasks[i] for i in train_idx], [tasks[i] for i in test_idx]


def find_non_overlapping_splits(tasks: Sequence[Any], n_splits: int, start_seed: int,) -> Tuple[List[int], Dict[int, set]]:
    chosen_seeds: List[int] = []
    chosen_test_sets: Dict[int, set] = {}
    candidate = start_seed

    while len(chosen_seeds) < n_splits:
        _, test_idx, _, _ = build_split(tasks, candidate)
        test_set = set(test_idx.tolist())

        if not any(test_set & prev for prev in chosen_test_sets.values()):
            chosen_seeds.append(candidate)
            chosen_test_sets[candidate] = test_set

        candidate += 1

    return chosen_seeds, chosen_test_sets


def save_and_print_split_manifest(tasks: Sequence[Any], split_id: int, split_seed: int, train_idx: np.ndarray, test_idx: np.ndarray, manifest_dir: Path,) -> None:
    rows = []

    print("\n" + "-" * 100)
    print(f"SPLIT {split_id} | split_seed={split_seed}")
    print("TRAIN index -> goal/task data")

    for idx in train_idx.tolist():
        desc = describe_task_short(tasks[idx])
        print(f"  train {idx:02d}: {desc}")
        row = {
            "split_id": split_id,
            "split_seed": split_seed,
            "group": "train",
            "task_index": idx,
            **decode_task_data(tasks[idx]),
        }
        rows.append(row)

    print("TEST index -> goal/task data")
    for idx in test_idx.tolist():
        desc = describe_task_short(tasks[idx])
        print(f"  test  {idx:02d}: {desc}")
        row = {
            "split_id": split_id,
            "split_seed": split_seed,
            "group": "test",
            "task_index": idx,
            **decode_task_data(tasks[idx]),
        }
        rows.append(row)

    manifest_dir.mkdir(parents=True, exist_ok=True)

    json_path = manifest_dir / f"button-press_v3_split_{split_id}_seed_{split_seed}_manifest.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    csv_path = manifest_dir / f"button-press_v3_split_{split_id}_seed_{split_seed}_manifest.csv"
    compact_rows = []
    for r in rows:
        compact_rows.append(
            {
                "split_id": r["split_id"],
                "split_seed": r["split_seed"],
                "group": r["group"],
                "task_index": r["task_index"],
                "task_env_name": r.get("task_env_name", ""),
                "rand_vec": json.dumps(r.get("rand_vec", "")),
                "target_pos": json.dumps(r.get("target_pos", r.get("_target_pos", r.get("goal", "")))),
                "obj_init_pos": json.dumps(r.get("obj_init_pos", r.get("_obj_init_pos", ""))),
            }
        )
    pd.DataFrame(compact_rows).to_csv(csv_path, index=False)

    print(f"Saved split manifest: {json_path}")
    print(f"Saved split CSV:      {csv_path}")


@dataclass
class EvalMetrics:
    success_rate: float
    avg_return: float
    std_return: float
    avg_steps: float
    std_steps: float
    avg_first_success_step: float
    episodes: int


def normalize_obs_for_model(vecnorm: Optional[VecNormalize], obs: np.ndarray) -> np.ndarray:
    if vecnorm is None:
        return obs
    obs_batch = np.asarray(obs, dtype=np.float32).reshape(1, -1)
    return vecnorm.normalize_obs(obs_batch)


def evaluate_on_task_list(
    model: PPO,
    vecnorm: Optional[VecNormalize],
    tasks: Sequence[Any],
    env_seed: int,
    n_episodes_per_task: int,
    deterministic: bool,
    terminate_on_success: bool,
    reward_type: str,
    max_episode_steps: int,
    group_name: str,
    config_name: str,
    split_id: int,
    split_seed: int,
    output_rows: List[Dict[str, Any]],
) -> EvalMetrics:
    env = make_mt1_env(
        seed=env_seed,
        tasks=list(tasks),
        sample_on_reset=False,
        terminate_on_success=terminate_on_success,
        render_mode=None,
        reward_type=reward_type,
        max_episode_steps=max_episode_steps,
    )

    base_env = env.unwrapped

    successes: List[float] = []
    returns: List[float] = []
    steps_all: List[int] = []
    first_success_steps: List[float] = []

    for task_local_idx, task in enumerate(tasks):
        sys.stdout.write(f"\r    {group_name}: task {task_local_idx + 1}/{len(tasks)} ...")
        sys.stdout.flush()

        base_env.set_task(task)

        for ep in range(n_episodes_per_task):
            obs, _ = env.reset()
            done = False
            ep_return = 0.0
            ep_steps = 0
            ep_success = 0.0
            first_success_step = np.nan

            while not done:
                model_obs = normalize_obs_for_model(vecnorm, obs)
                action, _ = model.predict(model_obs, deterministic=deterministic)

                # If obs was batched for VecNormalize, action is batched too.
                if isinstance(action, np.ndarray) and action.ndim > 1:
                    action = action[0]

                obs, reward, terminated, truncated, info = env.step(action)
                ep_return += float(reward)
                ep_steps += 1

                current_success = float(info.get("success", 0.0))
                if current_success > 0.0:
                    ep_success = 1.0
                    if np.isnan(first_success_step):
                        first_success_step = ep_steps

                done = bool(terminated or truncated)

            successes.append(ep_success)
            returns.append(ep_return)
            steps_all.append(ep_steps)
            first_success_steps.append(first_success_step)

            output_rows.append(
                {
                    "config": config_name,
                    "split_id": split_id,
                    "split_seed": split_seed,
                    "group": group_name,
                    "task_local_idx": task_local_idx,
                    "episode": ep,
                    "success": ep_success,
                    "return": ep_return,
                    "steps": ep_steps,
                    "first_success_step": first_success_step,
                }
            )

    env.close()
    sys.stdout.write("\r" + " " * 80 + "\r")
    sys.stdout.flush()

    finite_first_steps = [x for x in first_success_steps if not np.isnan(x)]

    return EvalMetrics(
        success_rate=float(np.mean(successes)) if successes else 0.0,
        avg_return=float(np.mean(returns)) if returns else 0.0,
        std_return=float(np.std(returns)) if returns else 0.0,
        avg_steps=float(np.mean(steps_all)) if steps_all else 0.0,
        std_steps=float(np.std(steps_all)) if steps_all else 0.0,
        avg_first_success_step=float(np.mean(finite_first_steps)) if finite_first_steps else float("nan"),
        episodes=len(successes),
    )


def create_vec_env(
    tasks: Sequence[Any],
    n_envs: int,
    base_seed: int,
    terminate_on_success: bool,
    reward_type: str,
    max_episode_steps: int,
    use_vecnormalize: bool,
    gamma: float,
    start_method: Optional[str],
):
    env_fns = [
        make_train_env_fn(
            rank=i,
            base_seed=base_seed,
            tasks=tasks,
            terminate_on_success=terminate_on_success,
            reward_type=reward_type,
            max_episode_steps=max_episode_steps,
        )
        for i in range(n_envs)
    ]

    vec_env = SubprocVecEnv(env_fns, start_method=start_method)
    vec_env = VecMonitor(vec_env)

    if use_vecnormalize:
        vec_env = VecNormalize(
            vec_env,
            norm_obs=True,
            norm_reward=True,
            clip_obs=10.0,
            clip_reward=10.0,
            gamma=gamma,
        )

    return vec_env


# =============================================================================
# Main
# =============================================================================


def main() -> None:

    args = SimpleNamespace(
        total_timesteps=RUN_TOTAL_TIMESTEPS,
        n_envs=RUN_N_ENVS,
        n_splits=RUN_N_SPLITS,
        task_seed=RUN_TASK_SEED,
        start_split_seed=RUN_START_SPLIT_SEED,
        train_seeds=",".join(str(s) for s in RUN_TRAIN_SEEDS),
        configs=RUN_CONFIGS,
        output_dir=RUN_OUTPUT_DIR,
        checkpoint_freq=RUN_CHECKPOINT_FREQ,
        reward_type=RUN_REWARD_TYPE,
        max_episode_steps=RUN_MAX_EPISODE_STEPS,
        device=RUN_DEVICE,
        start_method=RUN_START_METHOD,
        no_vecnormalize=not RUN_USE_VECNORMALIZE,
        train_terminate_on_success=RUN_TRAIN_TERMINATE_ON_SUCCESS,
        eval_terminate_on_success=RUN_EVAL_TERMINATE_ON_SUCCESS,
        no_progress_bar=not RUN_SHOW_PROGRESS_BAR,
        overwrite=RUN_OVERWRITE_EXISTING,
        eval_train_episodes_per_task=RUN_EVAL_TRAIN_EPISODES_PER_TASK,
        eval_test_episodes_per_task=RUN_EVAL_TEST_EPISODES_PER_TASK,
        eval_seed=RUN_EVAL_SEED,
    )

    selected_configs = args.configs
    bad_configs = [c for c in selected_configs if c not in PPO_CONFIGS]
    if bad_configs:
        raise ValueError(f"Unknown configs: {bad_configs}. Available: {list(PPO_CONFIGS.keys())}")

    train_seeds = parse_int_list(args.train_seeds)
    use_vecnormalize = not args.no_vecnormalize
    progress_bar = not args.no_progress_bar

    output_dir = ensure_dir(args.output_dir)
    model_dir = ensure_dir(output_dir / "models")
    checkpoint_dir = ensure_dir(output_dir / "checkpoints")
    tb_dir = ensure_dir(output_dir / "tensorboard")
    result_dir = ensure_dir(output_dir / "results")
    manifest_dir = ensure_dir(output_dir / "manifests")

    aggregate_csv = result_dir / "button-press_v3_aggregate_results.csv"
    per_episode_csv = result_dir / "button-press_v3_per_episode_results.csv"
    skipped_csv = result_dir / "button-press_v3_skipped_runs.csv"

    print("=" * 110)
    print("button-press-V3 PPO TRAINING")
    print("=" * 110)
    print(f"Environment: {ENV_NAME}")
    print(f"Total timesteps/model: {args.total_timesteps:,}")
    print(f"SubprocVecEnv workers: {args.n_envs}")
    print(f"Splits: {args.n_splits} non-overlapping 45/5 splits")
    print(f"Train seeds: {train_seeds}")
    print(f"Configs: {selected_configs}")
    print(f"VecNormalize: {use_vecnormalize}")
    print(f"Training terminate_on_success: {args.train_terminate_on_success}")
    print(f"Eval terminate_on_success: {args.eval_terminate_on_success}")
    print(f"Checkpoint frequency: {args.checkpoint_freq:,} env steps")
    print("=" * 110)

    print("\nExtracting 50 button-press-v3 task variations ...")
    all_tasks = extract_all_tasks(args.task_seed)

    print(f"Finding {args.n_splits} non-overlapping 45/5 splits ...")
    split_seeds, test_sets = find_non_overlapping_splits(
        all_tasks,
        n_splits=args.n_splits,
        start_seed=args.start_split_seed,
    )

    for split_id, split_seed in enumerate(split_seeds):
        train_idx, test_idx, _, _ = build_split(all_tasks, split_seed)
        print(f"  split_id={split_id} | split_seed={split_seed} | test_idx={sorted(test_sets[split_seed])}")
        save_and_print_split_manifest(
            tasks=all_tasks,
            split_id=split_id,
            split_seed=split_seed,
            train_idx=train_idx,
            test_idx=test_idx,
            manifest_dir=manifest_dir,
        )

    total_runs = len(split_seeds) * len(selected_configs) * len(train_seeds)
    run_count = 0

    for split_id, split_seed in enumerate(split_seeds):
        train_idx, test_idx, train_tasks, test_tasks = build_split(all_tasks, split_seed)

        for config_name in selected_configs:
            config = dict(PPO_CONFIGS[config_name])
            rollout_size = config["n_steps"] * args.n_envs

            for train_seed in train_seeds:
                run_count += 1
                run_name = (
                    f"ppo_{ENV_NAME}_{config_name}"
                    f"_split{split_id}_splitseed{split_seed}"
                    f"_trainseed{train_seed}_{args.total_timesteps}"
                )
                run_model_dir = ensure_dir(model_dir / run_name)
                run_checkpoint_dir = ensure_dir(checkpoint_dir / run_name)

                final_model_path = run_model_dir / f"{run_name}_final.zip"
                final_vecnorm_path = run_model_dir / f"{run_name}_vecnormalize.pkl"
                config_path = run_model_dir / f"{run_name}_config.json"

                print("\n" + "=" * 110)
                print(f"RUN {run_count}/{total_runs}: {run_name}")
                print("=" * 110)
                print(
                    f"PPO params: n_steps={config['n_steps']} | rollout_size={rollout_size} | "
                    f"batch_size={config['batch_size']} | lr={config['learning_rate']} | "
                    f"gamma={config['gamma']} | ent_coef={config['ent_coef']}"
                )

                if final_model_path.exists() and not args.overwrite:
                    print(f"Skipping existing model: {final_model_path}")
                    append_csv_row(
                        skipped_csv,
                        {
                            "run_name": run_name,
                            "reason": "final_model_exists",
                            "model_path": str(final_model_path),
                        },
                    )
                    continue

                with config_path.open("w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "env_name": ENV_NAME,
                            "config_name": config_name,
                            "ppo_config": config,
                            "split_id": split_id,
                            "split_seed": split_seed,
                            "train_seed": train_seed,
                            "train_idx": train_idx.tolist(),
                            "test_idx": test_idx.tolist(),
                            "n_envs": args.n_envs,
                            "total_timesteps": args.total_timesteps,
                            "vecnormalize": use_vecnormalize,
                            "train_terminate_on_success": args.train_terminate_on_success,
                            "eval_terminate_on_success": args.eval_terminate_on_success,
                        },
                        f,
                        indent=2,
                    )

                vec_env = create_vec_env(
                    tasks=train_tasks,
                    n_envs=args.n_envs,
                    base_seed=args.task_seed + 1000 * split_id + 100 * train_seed,
                    terminate_on_success=args.train_terminate_on_success,
                    reward_type=args.reward_type,
                    max_episode_steps=args.max_episode_steps,
                    use_vecnormalize=use_vecnormalize,
                    gamma=float(config["gamma"]),
                    start_method=args.start_method,
                )

                save_freq_calls = max(1, args.checkpoint_freq // args.n_envs)
                checkpoint_callback = CheckpointCallback(
                    save_freq=save_freq_calls,
                    save_path=str(run_checkpoint_dir),
                    name_prefix=run_name,
                    save_replay_buffer=False,
                    save_vecnormalize=use_vecnormalize,
                )
                callbacks = CallbackList([checkpoint_callback])

                model = PPO(
                    policy="MlpPolicy",
                    env=vec_env,
                    verbose=0,
                    seed=train_seed,
                    device=args.device,
                    tensorboard_log=str(tb_dir),
                    **config,
                )

                try:
                    model.learn(
                        total_timesteps=args.total_timesteps,
                        tb_log_name=run_name,
                        callback=callbacks,
                        progress_bar=progress_bar,
                    )

                    model.save(str(final_model_path))
                    if use_vecnormalize:
                        vec_env.save(str(final_vecnorm_path))

                    # Freeze normalization stats for manual eval.
                    eval_vecnorm: Optional[VecNormalize] = None
                    if use_vecnormalize:
                        vec_env.training = False
                        vec_env.norm_reward = False
                        eval_vecnorm = vec_env

                    print("\nEvaluating final model on TRAIN tasks ...")
                    per_episode_rows: List[Dict[str, Any]] = []
                    train_metrics = evaluate_on_task_list(
                        model=model,
                        vecnorm=eval_vecnorm,
                        tasks=train_tasks,
                        env_seed=args.eval_seed + split_id,
                        n_episodes_per_task=args.eval_train_episodes_per_task,
                        deterministic=True,
                        terminate_on_success=args.eval_terminate_on_success,
                        reward_type=args.reward_type,
                        max_episode_steps=args.max_episode_steps,
                        group_name="train",
                        config_name=config_name,
                        split_id=split_id,
                        split_seed=split_seed,
                        output_rows=per_episode_rows,
                    )

                    print("Evaluating final model on TEST tasks ...")
                    test_metrics = evaluate_on_task_list(
                        model=model,
                        vecnorm=eval_vecnorm,
                        tasks=test_tasks,
                        env_seed=args.eval_seed + 10_000 + split_id,
                        n_episodes_per_task=args.eval_test_episodes_per_task,
                        deterministic=True,
                        terminate_on_success=args.eval_terminate_on_success,
                        reward_type=args.reward_type,
                        max_episode_steps=args.max_episode_steps,
                        group_name="test",
                        config_name=config_name,
                        split_id=split_id,
                        split_seed=split_seed,
                        output_rows=per_episode_rows,
                    )

                    for row in per_episode_rows:
                        row.update(
                            {
                                "run_name": run_name,
                                "train_seed": train_seed,
                                "total_timesteps": args.total_timesteps,
                                "model_path": str(final_model_path),
                            }
                        )
                        append_csv_row(per_episode_csv, row)

                    success_gap = train_metrics.success_rate - test_metrics.success_rate
                    return_gap = train_metrics.avg_return - test_metrics.avg_return

                    aggregate_row = {
                        "run_name": run_name,
                        "env_name": ENV_NAME,
                        "config_name": config_name,
                        "split_id": split_id,
                        "split_seed": split_seed,
                        "train_seed": train_seed,
                        "total_timesteps": args.total_timesteps,
                        "n_envs": args.n_envs,
                        "rollout_size": rollout_size,
                        "vecnormalize": use_vecnormalize,
                        "train_idx": train_idx.tolist(),
                        "test_idx": test_idx.tolist(),
                        "train_success_rate": train_metrics.success_rate,
                        "train_avg_return": train_metrics.avg_return,
                        "train_std_return": train_metrics.std_return,
                        "train_avg_steps": train_metrics.avg_steps,
                        "train_avg_first_success_step": train_metrics.avg_first_success_step,
                        "train_episodes": train_metrics.episodes,
                        "test_success_rate": test_metrics.success_rate,
                        "test_avg_return": test_metrics.avg_return,
                        "test_std_return": test_metrics.std_return,
                        "test_avg_steps": test_metrics.avg_steps,
                        "test_avg_first_success_step": test_metrics.avg_first_success_step,
                        "test_episodes": test_metrics.episodes,
                        "success_gap": success_gap,
                        "return_gap": return_gap,
                        "model_path": str(final_model_path),
                        "vecnormalize_path": str(final_vecnorm_path) if use_vecnormalize else "",
                        "tensorboard_dir": str(tb_dir),
                    }
                    append_csv_row(aggregate_csv, aggregate_row)

                    print("\nRESULT")
                    print("-" * 100)
                    print(f"Train success: {train_metrics.success_rate:.4f}")
                    print(f"Test success:  {test_metrics.success_rate:.4f}")
                    print(f"Success gap:   {success_gap:.4f}")
                    print(f"Train return:  {train_metrics.avg_return:.4f}")
                    print(f"Test return:   {test_metrics.avg_return:.4f}")
                    print(f"Return gap:    {return_gap:.4f}")
                    print(f"Final model:   {final_model_path}")
                    if use_vecnormalize:
                        print(f"VecNormalize:  {final_vecnorm_path}")
                    print(f"Aggregate CSV: {aggregate_csv}")
                    print(f"Per-ep CSV:    {per_episode_csv}")

                finally:
                    vec_env.close()

    print("\n" + "=" * 110)
    print("ALL DONE")
    print("=" * 110)
    print(f"Models:      {model_dir}")
    print(f"Checkpoints: {checkpoint_dir}")
    print(f"TensorBoard: tensorboard --logdir {tb_dir}")
    print(f"Results:     {result_dir}")


if __name__ == "__main__":
    main()
