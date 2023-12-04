import logging
from pathlib import Path
import random
from math import sqrt
from typing import Optional, Sequence, Union, Tuple
import gym
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from imitate.models import MpcModelBase, InvertedPendulumModel, CartpoleModel
from imitate.mpc_acados import MPCBaseClass

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def policy_rollout(policy, model: MpcModelBase, x0: np.ndarray, max_horizon: int):
    def violation(var, low, high):
        return np.abs(var - np.clip(var, low, high)).sum()

    tracking_costs = [0]
    violation_costs = [0]
    violations = [0]
    x = np.zeros((max_horizon + 1, model.ns))
    u = np.zeros((max_horizon, model.na))
    x[0, :] = x0
    for k in range(max_horizon):
        u[k, :] = policy(x[k, :])
        x_tmp, cost_tmp = model.integrator(x[k, :], u[k, :])
        x[k + 1, :], cost = x_tmp.full().squeeze(), cost_tmp.full().squeeze()
        tracking_costs[-1] += cost
        violation_costs[-1] += model.constr_cost_fn(x[k, :], u[k, :]).full().squeeze()
        # Calculate constraint violations
        v = violation(x[k, :], model.s_min, model.s_max)
        v += violation(u[k, :], model.a_min, model.a_max)
        violations[-1] += v
    v = violation(x[k + 1, :], model.s_min, model.s_max)
    violations[-1] += v

    return x.squeeze(), u.squeeze(), tracking_costs, violation_costs, violations


def evaluate_policy(
    policy, model: MpcModelBase, num_episodes=5000, max_horizon=20, seed=0, samples=None
):
    if isinstance(samples, Tuple):
        samples = samples[0]

    tracking_costs = []
    violation_costs = []
    violations = []
    if samples is None:
        # Sample initial states
        rng = np.random.default_rng(seed)
        for _ in tqdm(range(num_episodes), desc="Evaluating policy"):
            x = rng.uniform(low=0.2 * model.s_min, high=0.2 * model.s_max)
            _, _, t_cost, v_cost, viol = policy_rollout(policy, model, x, max_horizon)
            tracking_costs += t_cost
            violation_costs += v_cost
            violations += viol
        # avg_violation = np.mean(np.array(violations)/(num_episodes * max_horizon))
        return {
            "tracking_costs": np.array(tracking_costs),
            "violation_costs": np.array(violation_costs),
            "violations": np.array(violations),
        }
    else:
        # Initial states are given
        if samples.shape[0] < num_episodes:
            logger.warning(
                "The available samples are less than the number of desired episodes!"
            )
        for x in tqdm(samples, desc="Evaluating policy"):
            _, _, t_cost, v_cost, viol = policy_rollout(policy, model, x, max_horizon)
            tracking_costs += t_cost
            violation_costs += v_cost
            violations += viol
        # avg_violation = np.mean(np.array(violations)/(num_episodes * max_horizon))
        return {
            "tracking_costs": np.array(tracking_costs),
            "violation_costs": np.array(violation_costs),
            "violation_bool": np.array(violations).astype(bool),
            }

def log_eval_stats(eval_stats_dict: dict):
        logger.info(f"Mean tracking cost: {eval_stats_dict['tracking_costs'].mean():.3f} +- {eval_stats_dict['tracking_costs'].std():.3f}")
        logger.info(f"Mean violation cost: {eval_stats_dict['violation_costs'].mean():.3f} +- {eval_stats_dict['violation_costs'].std():.3f}")
        logger.info(f"Violations: {eval_stats_dict['violation_bool'].mean():.3f}")

def save_eval_stats_to_csv(eval_stats_dict:dict, exp_dir:Path, filename:str = 'performance_avg.csv'):
    with open(exp_dir / filename, "w") as f:
        f.write(
            f"mean_track_cost,std_track_cost,mean_viol_cost,std_viol_cost,violation\n"
            f"{eval_stats_dict['tracking_costs'].mean()},"
            f"{eval_stats_dict['tracking_costs'].std()},"
            f"{eval_stats_dict['violation_costs'].mean()},"
            f"{eval_stats_dict['violation_costs'].std()},"
            f"{eval_stats_dict['violation_bool'].mean()}"
        )


def save_full_eval_stats_to_csv(
    eval_stats_dict: dict, exp_dir: Path, filename: str = "performance_full.csv"
):
    to_save = np.vstack(
        [
            eval_stats_dict["tracking_costs"],
            eval_stats_dict["violation_costs"],
            eval_stats_dict["violation_bool"],
        ]
    )
    df = pd.DataFrame(to_save.T, columns=list(eval_stats_dict.keys()))
    df.to_csv(exp_dir / filename, index=False)


def set_seed(seed: Optional[int]):
    """Setting seed to make runs reproducible.
    Args:
        seed: The seed to set.
    """
    if seed is None:
        return
    logger.info(f"Set global seed to {seed}")
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def latexify(fig_width=None, fig_height=None):
    """Set up matplotlib's RC params for LaTeX plotting.
    Call this before plotting a figure.

    Parameters
    ----------
    fig_width : float, optional, inches
    fig_height : float,  optional, inches
    columns : {1, 2}
    """

    # code adapted from http://www.scipy.org/Cookbook/Matplotlib/LaTeX_Examples

    # Width and max height in inches for IEEE journals taken from
    # computer.org/cms/Computer.org/Journal%20templates/transactions_art_guide.pdf

    if fig_width is None:
        fig_width = 5  # width in inches

    if fig_height is None:
        golden_mean = (sqrt(5) - 1.0) / 2.0  # Aesthetic ratio
        fig_height = fig_width * golden_mean  # height in inches

    params = {
        # "backend": "ps",
        "text.latex.preamble": r"\usepackage{gensymb} \usepackage{amsmath}",
        "axes.labelsize": 10,  # fontsize for x and y labels (was 10)
        "axes.titlesize": 10,
        "lines.linewidth": 2,
        "legend.fontsize": 10,  # was 10
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "text.usetex": True,
        "figure.figsize": [fig_width, fig_height],
        "font.family": "serif",
    }

    matplotlib.rcParams.update(params)


def plot_trajectory(
    s_collection: Union[np.ndarray, list],
    a_collection: Union[np.ndarray, list],
    model: MpcModelBase,
    title: str,
):

    alpha = 0.4
    if isinstance(a_collection, np.ndarray):
        a_collection = [a_collection]
        alpha = 0.7
    if isinstance(s_collection, np.ndarray):
        s_collection = [s_collection]
    N = a_collection[0].shape[0]
    dt = model.dt
    time_array = np.linspace(0, N * dt, N + 1)
    # latexify()

    fig, axs = plt.subplots(model.ns + model.na, 1, figsize=(4.5, 8), sharex=True)

    for s in range(model.ns):
        axs[s].grid()
        for s_traj in s_collection:
            if len(s_traj.shape) == 1:
                s_traj = s_traj[..., np.newaxis]
            axs[s].plot(time_array, s_traj[:, s], "-", alpha=alpha, color="tab:blue")
        axs[s].axhline(model.s_max[s], linestyle=":", color="k", alpha=0.7)
        axs[s].axhline(model.s_min[s], linestyle=":", color="k", alpha=0.7)
    for a in range(model.na):
        for a_traj in a_collection:
            if len(a_traj.shape) == 1:
                a_traj = a_traj[..., np.newaxis]
            axs[-1].step(
                time_array,
                model.scaling_factor_action * np.append([a_traj[0, a]], a_traj[:, a]),
                alpha=alpha,
                color="tab:orange",
            )
    axs[0].set_title(title)
    axs[-1].grid()
    axs[-1].axhline(
        model.scaling_factor_action * model.a_max, linestyle=":", color="k", alpha=0.5
    )
    axs[-1].axhline(
        model.scaling_factor_action * model.a_min, linestyle=":", color="k", alpha=0.5
    )
    axs[-1].set_xlabel(r"time [sec]")

    if model.name == "inverted_pendulum":
        axs[0].set_ylabel(r"$\theta \; \mathrm{[rad]}$")
        axs[1].set_ylabel(r"$\omega \; \mathrm{[rad/s]}$")
        axs[-1].set_ylabel(r"$a \; \mathrm{[rad/sec^2]}$")
    if model.name == "cartpole":
        axs[0].set_ylabel(r"$p \; \mathrm{[m]}$")
        axs[1].set_ylabel(r"$v \; \mathrm{[m/s]}$")
        axs[2].set_ylabel(r"$\theta \; \mathrm{[rad]}$")
        axs[3].set_ylabel(r"$\omega \; \mathrm{[rad/s]}$")
        axs[-1].set_ylabel(r"$a \; \mathrm{[rad/sec^2]}$")
        for idx, ax in enumerate(axs):
            ax.get_yaxis().set_label_coords(-0.07, 0.5)
            if idx < model.ns:
                try:
                    ax.set_ylim(2 * model.s_min[idx], 2 * model.s_max[idx])
                except:
                    pass
            else:
                try:
                    ax.set_ylim(
                        2 * model.scaling_factor_action * model.a_min[idx - model.ns],
                        2 * model.scaling_factor_action * model.a_max[idx - model.ns],
                    )
                except:
                    pass

    # plt.tight_layout()
    plt.subplots_adjust(hspace=0.15)


class DaggerWrapper(gym.Env):
    def __init__(self, mpc: MPCBaseClass, N: int = 50):
        super().__init__()
        self.mpc = mpc

        self.N = N
        self.n = 0
        self.state = None

    def expert(self):
        sol = self.mpc.solve_ocp(self.state)
        return (
            sol["controls"][0, :],
            sol["value_cost_fn"],
            sol["solver_stats"]["success"],
            {"states": sol["states"], "controls": sol["controls"], "slacks_lower": sol["slacks"]["lower"], "slacks_upper": sol["slacks"]["upper"]}, #TODO: casadi/acados
        )

    def reset(self):
        self.state = self.mpc.collect_initial_state(1, feasible=True)
        self.n = 0
        return self.state

    def step(self, action):
        state, cost = self.mpc.model.integrator(self.state, action)
        self.state = state.full().T
        self.n += 1
        done = True if self.n == self.N else False

        return self.state, cost.full(), done, {}


def collect_samples_dagger(
    env: DaggerWrapper, policy, num_episodes=50, fraction_expert=0.3
):
    states = []
    optimal_u = []
    optimal_obj = []
    done = True
    traj_linearization = []
    for i in tqdm(range(num_episodes), desc="Collecting samples"):
        if done:
            state = env.reset()

        use_expert = True if random.random() < fraction_expert else False
        expert_u, expert_obj, ocp_success, lin_vec = env.expert()
        if ocp_success:
            policy_u = policy(state)
            dagger_u = expert_u if use_expert else policy_u

            states.append(state.squeeze())
            optimal_u.append(expert_u)
            optimal_obj.append(expert_obj)
            traj_linearization.append(lin_vec)

            state, _, done, _ = env.step(dagger_u)
        else:
            done = True
            logger.warning(f"Shorter Dagger rollout: {i}/{num_episodes} - solver failed")

    return states, optimal_u, optimal_obj, traj_linearization


def create_fc_net(input_dim: int, output_dim: int, hidden_dims: Sequence[int], tanh):
    layers = []
    for i, hidden_dim in enumerate(hidden_dims):
        if i == 0:
            layers.append(nn.Linear(input_dim, hidden_dim))
        else:
            layers.append(nn.Linear(hidden_dims[i - 1], hidden_dim))
        layers.append(nn.ReLU())
    layers.append(nn.Linear(hidden_dims[-1], output_dim))

    if tanh:
        layers.append(nn.Tanh())
    return nn.Sequential(*layers)


def instanciate_model_object(dynamic_model: str):
    if dynamic_model == "inverted_pendulum":
        return InvertedPendulumModel()
    elif dynamic_model == "cartpole":
        return CartpoleModel()
