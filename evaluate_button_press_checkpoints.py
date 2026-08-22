from __future__ import annotations

import csv
import json
import os
import pickle
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import gymnasium as gym
import metaworld  # type: ignore
import numpy as np
import pandas as pd

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


ENV_ID = "Meta-World/MT1"
ENV_NAME = "button-press-v3"

ROOT_DIR = Path("./button-press_v3_ppo_split_runs")
OUTPUT_DIR = ROOT_DIR / "checkpoint_evaluation_all"

# None = evaluate every config found in checkpoint config.json files
CONFIGS_TO_EVALUATE = None

GROUPS_TO_EVALUATE = ["train", "test"]

EVAL_TRAIN_EPISODES_PER_TASK = 1
EVAL_TEST_EPISODES_PER_TASK = 1

EVAL_SEED = 1000
DETERMINISTIC = True
DEVICE = "cpu"

REWARD_TYPE = "v2"
MAX_EPISODE_STEPS = 500
TERMINATE_ON_SUCCESS = True
STOP_ON_SUCCESS = True

APPEND_TO_EXISTING = True

# None = evaluate every saved checkpoint.
# 100_000 = evaluate only checkpoints at 100k, 200k, 300k, ...
EVALUATE_ONLY_EVERY_N_STEPS = 5_000


class ProgressBar:
    BAR_WIDTH = 36

    def __init__(self, total: int, desc: str = ""):
        self.total = max(total, 1)
        self.desc = desc
        self.n = 0
        self.start = time.time()
        self.draw()

    def update(self, n: int = 1):
        self.n = min(self.n + n, self.total)
        self.draw()

    def draw(self):
        frac = self.n / self.total
        filled = int(self.BAR_WIDTH * frac)
        bar = "█" * filled + "░" * (self.BAR_WIDTH - filled)
        elapsed = time.time() - self.start
        if 0 < frac < 1:
            eta = elapsed / frac - elapsed
            eta_s = f" ETA {eta:6.0f}s"
        elif frac >= 1:
            eta_s = f" {elapsed:6.0f}s total"
        else:
            eta_s = ""
        sys.stdout.write(
            f"\r{self.desc} [{bar}] {frac * 100:5.1f}% "
            f"{self.n}/{self.total}{eta_s}   "
        )
        sys.stdout.flush()

    def close(self):
        self.n = self.total
        self.draw()
        sys.stdout.write("\n")
        sys.stdout.flush()


def write_rows(csv_path: Path, rows: List[Dict[str, Any]], append: bool = True) -> None:
    if not rows:
        return
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append and csv_path.exists() else "w"
    with csv_path.open(mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if mode == "w":
            writer.writeheader()
        writer.writerows(rows)


def get_task_wrapper(env: gym.Env):
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


def make_mt1_env(
    seed: int,
    tasks: Optional[Sequence[Any]] = None,
    sample_on_reset: bool = True,
    terminate_on_success: bool = False,
    render_mode: Optional[str] = None,
    reward_type: str = REWARD_TYPE,
    max_episode_steps: int = MAX_EPISODE_STEPS,
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


def extract_all_tasks(task_seed: int) -> List[Any]:
    env = make_mt1_env(seed=task_seed, tasks=None, sample_on_reset=True)
    wrapper = get_task_wrapper(env)
    tasks = list(wrapper.tasks)
    env.close()
    if len(tasks) != 50:
        raise RuntimeError(f"Expected 50 MT1 tasks, got {len(tasks)}")
    return tasks


def build_split_from_indices(tasks: Sequence[Any], train_idx: Sequence[int], test_idx: Sequence[int]):
    train_tasks = [tasks[int(i)] for i in train_idx]
    test_tasks = [tasks[int(i)] for i in test_idx]
    return train_tasks, test_tasks


@dataclass
class CheckpointItem:
    run_name: str
    config_name: str
    split_id: int
    split_seed: int
    train_seed: int
    total_timesteps: int
    task_seed: int
    train_idx: List[int]
    test_idx: List[int]
    checkpoint_step: int
    model_path: Path
    vecnormalize_path: Optional[Path]
    config_json_path: Path


def parse_checkpoint_step(path: Path) -> Optional[int]:
    m = re.search(r"_(\d+)_steps\.zip$", path.name)
    if m:
        return int(m.group(1))
    return None


def find_matching_vecnormalize(checkpoint_zip: Path, checkpoint_step: int) -> Optional[Path]:
    folder = checkpoint_zip.parent
    candidates = [
        folder / f"vecnormalize_{checkpoint_step}_steps.pkl",
        folder / f"{checkpoint_zip.stem}_vecnormalize.pkl",
        folder / f"{checkpoint_zip.stem}.pkl",
    ]
    for p in candidates:
        if p.exists():
            return p

    step_text = str(checkpoint_step)
    matches = sorted([
        p for p in folder.glob("*vecnormalize*.pkl")
        if step_text in p.name
    ])
    if matches:
        return matches[0]

    all_vec = sorted(folder.glob("*vecnormalize*.pkl"))
    if len(all_vec) == 1:
        return all_vec[0]

    return None


def discover_checkpoints(root_dir: Path) -> List[CheckpointItem]:
    checkpoint_root = root_dir / "checkpoints"
    model_root = root_dir / "models"

    if not checkpoint_root.exists():
        raise FileNotFoundError(f"Checkpoint folder not found: {checkpoint_root}")

    checkpoint_files = sorted(checkpoint_root.rglob("*.zip"))
    items: List[CheckpointItem] = []
    skipped = []

    for model_path in checkpoint_files:
        checkpoint_step = parse_checkpoint_step(model_path)
        if checkpoint_step is None:
            skipped.append({"model_path": str(model_path), "reason": "could_not_parse_checkpoint_step"})
            continue

        run_name = model_path.parent.name
        config_json_path = model_root / run_name / f"{run_name}_config.json"

        if not config_json_path.exists():
            skipped.append({
                "model_path": str(model_path),
                "reason": "missing_config_json",
                "expected_config_json": str(config_json_path),
            })
            continue

        with config_json_path.open("r", encoding="utf-8") as f:
            cfg = json.load(f)

        config_name = str(cfg["config_name"])
        if CONFIGS_TO_EVALUATE is not None and config_name not in CONFIGS_TO_EVALUATE:
            continue

        vecnormalize_path = find_matching_vecnormalize(model_path, checkpoint_step)

        items.append(
            CheckpointItem(
                run_name=run_name,
                config_name=config_name,
                split_id=int(cfg["split_id"]),
                split_seed=int(cfg["split_seed"]),
                train_seed=int(cfg["train_seed"]),
                total_timesteps=int(cfg["total_timesteps"]),
                task_seed=int(cfg.get("task_seed", 67)),
                train_idx=[int(x) for x in cfg["train_idx"]],
                test_idx=[int(x) for x in cfg["test_idx"]],
                checkpoint_step=checkpoint_step,
                model_path=model_path,
                vecnormalize_path=vecnormalize_path,
                config_json_path=config_json_path,
            )
        )

    if skipped:
        write_rows(OUTPUT_DIR / "checkpoint_discovery_skipped.csv", skipped, append=False)

    items.sort(key=lambda x: (x.config_name, x.split_id, x.train_seed, x.checkpoint_step))

    if EVALUATE_ONLY_EVERY_N_STEPS is not None:
        before = len(items)
        items = [
            item for item in items
            if item.checkpoint_step % int(EVALUATE_ONLY_EVERY_N_STEPS) == 0
        ]
        print(
            f"Filtered checkpoints with EVALUATE_ONLY_EVERY_N_STEPS="
            f"{EVALUATE_ONLY_EVERY_N_STEPS}: {before} -> {len(items)}"
        )

    return items


def load_vecnormalize_for_eval(vecnormalize_path: Optional[Path], tasks: Sequence[Any]) -> Optional[VecNormalize]:
    if vecnormalize_path is None or not vecnormalize_path.exists():
        return None

    dummy_env = DummyVecEnv([
        lambda: make_mt1_env(
            seed=999,
            tasks=list(tasks),
            sample_on_reset=False,
            terminate_on_success=TERMINATE_ON_SUCCESS,
            reward_type=REWARD_TYPE,
            max_episode_steps=MAX_EPISODE_STEPS,
        )
    ])

    vecnorm = VecNormalize.load(str(vecnormalize_path), dummy_env)
    vecnorm.training = False
    vecnorm.norm_reward = False
    return vecnorm


def normalize_obs(vecnorm: Optional[VecNormalize], obs: np.ndarray) -> np.ndarray:
    if vecnorm is None:
        return obs
    obs_batch = np.asarray(obs, dtype=np.float32).reshape(1, -1)
    return vecnorm.normalize_obs(obs_batch)


@dataclass
class EvalMetrics:
    success_rate: float
    avg_return: float
    std_return: float
    avg_steps: float
    std_steps: float
    avg_first_success_step: float
    episodes: int


def evaluate_checkpoint_on_tasks(
    model: PPO,
    vecnorm: Optional[VecNormalize],
    tasks: Sequence[Any],
    group_name: str,
    item: CheckpointItem,
    n_episodes_per_task: int,
    eval_seed: int,
) -> Tuple[EvalMetrics, List[Dict[str, Any]]]:

    env = make_mt1_env(
        seed=eval_seed,
        tasks=list(tasks),
        sample_on_reset=False,
        terminate_on_success=TERMINATE_ON_SUCCESS,
        reward_type=REWARD_TYPE,
        max_episode_steps=MAX_EPISODE_STEPS,
    )

    base_env = env.unwrapped

    rows: List[Dict[str, Any]] = []
    successes: List[float] = []
    returns: List[float] = []
    steps_all: List[int] = []
    first_success_steps: List[float] = []

    for task_local_idx, task in enumerate(tasks):
        base_env.set_task(task)

        for ep in range(n_episodes_per_task):
            obs, _ = env.reset(seed=eval_seed + ep + 1000 * task_local_idx)
            done = False
            ep_return = 0.0
            ep_steps = 0
            ep_success = 0.0
            first_success_step = np.nan

            while not done:
                model_obs = normalize_obs(vecnorm, obs)
                action, _ = model.predict(model_obs, deterministic=DETERMINISTIC)

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
                    if STOP_ON_SUCCESS:
                        done = True
                        break

                done = bool(terminated or truncated)

            successes.append(ep_success)
            returns.append(ep_return)
            steps_all.append(ep_steps)
            first_success_steps.append(first_success_step)

            rows.append(
                {
                    "run_name": item.run_name,
                    "config_name": item.config_name,
                    "split_id": item.split_id,
                    "split_seed": item.split_seed,
                    "train_seed": item.train_seed,
                    "checkpoint_step": item.checkpoint_step,
                    "total_timesteps": item.total_timesteps,
                    "group": group_name,
                    "task_local_idx": task_local_idx,
                    "episode": ep,
                    "success": ep_success,
                    "return": ep_return,
                    "steps": ep_steps,
                    "first_success_step": first_success_step,
                    "model_path": str(item.model_path),
                    "vecnormalize_path": str(item.vecnormalize_path) if item.vecnormalize_path else "",
                }
            )

    env.close()

    finite_first = [x for x in first_success_steps if not np.isnan(x)]

    metrics = EvalMetrics(
        success_rate=float(np.mean(successes)) if successes else 0.0,
        avg_return=float(np.mean(returns)) if returns else 0.0,
        std_return=float(np.std(returns)) if returns else 0.0,
        avg_steps=float(np.mean(steps_all)) if steps_all else 0.0,
        std_steps=float(np.std(steps_all)) if steps_all else 0.0,
        avg_first_success_step=float(np.mean(finite_first)) if finite_first else float("nan"),
        episodes=len(successes),
    )

    return metrics, rows


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    raw_csv = OUTPUT_DIR / "button_press_checkpoint_raw_episodes.csv"
    summary_csv = OUTPUT_DIR / "button_press_checkpoint_summary.csv"
    pivot_success_csv = OUTPUT_DIR / "button_press_checkpoint_success_pivot.csv"
    pivot_return_csv = OUTPUT_DIR / "button_press_checkpoint_return_pivot.csv"
    skipped_csv = OUTPUT_DIR / "button_press_checkpoint_skipped.csv"

    if not APPEND_TO_EXISTING:
        for p in [raw_csv, summary_csv, pivot_success_csv, pivot_return_csv, skipped_csv]:
            if p.exists():
                p.unlink()

    print("=" * 110)
    print("BUTTON-PRESS CHECKPOINT EVALUATION")
    print("=" * 110)
    print(f"Root dir: {ROOT_DIR}")
    print(f"Output dir: {OUTPUT_DIR}")
    print(f"Groups: {GROUPS_TO_EVALUATE}")
    print(f"Train episodes/task: {EVAL_TRAIN_EPISODES_PER_TASK}")
    print(f"Test episodes/task:  {EVAL_TEST_EPISODES_PER_TASK}")
    print(f"Stop on success: {STOP_ON_SUCCESS}")
    print("=" * 110)

    items = discover_checkpoints(ROOT_DIR)
    print(f"Found {len(items)} checkpoints.")
    if not items:
        print("No checkpoints found. Check ROOT_DIR and checkpoint naming.")
        return

    all_tasks_by_seed: Dict[int, List[Any]] = {}
    summary_rows: List[Dict[str, Any]] = []
    skipped_rows: List[Dict[str, Any]] = []

    pbar = ProgressBar(len(items), desc="Evaluating checkpoints")

    for item in items:
        try:
            if item.task_seed not in all_tasks_by_seed:
                all_tasks_by_seed[item.task_seed] = extract_all_tasks(item.task_seed)

            all_tasks = all_tasks_by_seed[item.task_seed]
            train_tasks, test_tasks = build_split_from_indices(all_tasks, item.train_idx, item.test_idx)

            model = PPO.load(str(item.model_path), device=DEVICE)
            vecnorm = load_vecnormalize_for_eval(item.vecnormalize_path, train_tasks)

            if item.vecnormalize_path is None:
                print(f"\nWarning: no VecNormalize found for {item.model_path}. Evaluating without normalization.")

            for group in GROUPS_TO_EVALUATE:
                if group == "train":
                    tasks = train_tasks
                    n_eps = EVAL_TRAIN_EPISODES_PER_TASK
                    seed = EVAL_SEED + item.split_id
                elif group == "test":
                    tasks = test_tasks
                    n_eps = EVAL_TEST_EPISODES_PER_TASK
                    seed = EVAL_SEED + 10_000 + item.split_id
                else:
                    raise ValueError(f"Unknown group: {group}")

                metrics, raw_rows = evaluate_checkpoint_on_tasks(
                    model=model,
                    vecnorm=vecnorm,
                    tasks=tasks,
                    group_name=group,
                    item=item,
                    n_episodes_per_task=n_eps,
                    eval_seed=seed,
                )

                write_rows(raw_csv, raw_rows, append=True)

                summary_rows.append(
                    {
                        "run_name": item.run_name,
                        "config_name": item.config_name,
                        "split_id": item.split_id,
                        "split_seed": item.split_seed,
                        "train_seed": item.train_seed,
                        "checkpoint_step": item.checkpoint_step,
                        "total_timesteps": item.total_timesteps,
                        "group": group,
                        "success_rate": metrics.success_rate,
                        "avg_return": metrics.avg_return,
                        "std_return": metrics.std_return,
                        "avg_steps": metrics.avg_steps,
                        "std_steps": metrics.std_steps,
                        "avg_first_success_step": metrics.avg_first_success_step,
                        "episodes": metrics.episodes,
                        "model_path": str(item.model_path),
                        "vecnormalize_path": str(item.vecnormalize_path) if item.vecnormalize_path else "",
                    }
                )

            if vecnorm is not None:
                vecnorm.close()

        except Exception as exc:
            skipped_rows.append(
                {
                    "run_name": item.run_name,
                    "config_name": item.config_name,
                    "split_id": item.split_id,
                    "train_seed": item.train_seed,
                    "checkpoint_step": item.checkpoint_step,
                    "model_path": str(item.model_path),
                    "reason": repr(exc),
                }
            )
            write_rows(skipped_csv, [skipped_rows[-1]], append=True)

        pbar.update(1)

    pbar.close()

    summary_df = pd.DataFrame(summary_rows)
    if summary_df.empty:
        print("No successful evaluations. See skipped CSV.")
        return

    summary_df = summary_df.sort_values(
        ["config_name", "split_id", "train_seed", "checkpoint_step", "group"]
    )

    # When APPEND_TO_EXISTING=True, keep old config_A-D rows and append config_E rows.
    # When APPEND_TO_EXISTING=False, this writes a fresh summary file.
    write_header = not summary_csv.exists()
    summary_df.to_csv(summary_csv, mode="a", header=write_header, index=False)

    # Re-read the full summary so pivot tables include both old configs and config_E.
    combined_summary_df = pd.read_csv(summary_csv)

    success_pivot = combined_summary_df.pivot_table(
        index=["checkpoint_step"],
        columns=["config_name", "group"],
        values="success_rate",
        aggfunc="mean",
    )
    success_pivot.to_csv(pivot_success_csv)

    return_pivot = combined_summary_df.pivot_table(
        index=["checkpoint_step"],
        columns=["config_name", "group"],
        values="avg_return",
        aggfunc="mean",
    )
    return_pivot.to_csv(pivot_return_csv)

    print("\n" + "=" * 110)
    print("DONE")
    print("=" * 110)
    print("Summary CSV:      ", summary_csv)
    print("Raw episodes CSV: ", raw_csv)
    print("Success pivot CSV:", pivot_success_csv)
    print("Return pivot CSV: ", pivot_return_csv)
    print("Skipped CSV:      ", skipped_csv)

    display_cols = ["config_name", "group", "checkpoint_step", "success_rate", "avg_return", "episodes"]
    print("\nFirst rows:")
    print(combined_summary_df[display_cols].head(40).to_string(index=False))


if __name__ == "__main__":
    main()
