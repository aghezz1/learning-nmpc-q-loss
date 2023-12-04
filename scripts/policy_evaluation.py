"""
    Learning Nonlinear MPC with Q-loss
    Imitation Learning from Nonlinear MPC via the Exact Q-Loss and its
    Gauss-Newton Approximation, IEEE CDC Conference 2023
    Copyright (C) 2023  Andrea Ghezzi, Jasper Hoffman, Jonathan Frey
    University of Freiburg, Germany (andrea.ghezzi@imtek.uni-freiburg.de)

    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License along
    with this program; if not, write to the Free Software Foundation, Inc.,
    51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
"""

from pathlib import Path
import time
import pandas as pd
import logging
from pathlib import Path
import pickle
import torch
from tqdm.contrib.concurrent import process_map
import yaml
from imitate.mpc_acados import MPCBaseClass
from imitate.utils import (
    create_fc_net,
    evaluate_policy,
    log_eval_stats,
    save_eval_stats_to_csv,
    save_full_eval_stats_to_csv,
    instanciate_model_object,
)

RESULTS_DIR = Path("results")
DATA_DIR = Path("experiments/cartpole/hp_sweep")
SAMPLES_DIR = Path("samples")

QUANTILE = 0.99
ALPHA_SPAN = 0.3
FEASIBLE = True
TANH = True
SAMPLE_NAME = (
    f"cartpole_eval_samples_feasible_{FEASIBLE}_constraint_span_{ALPHA_SPAN}_seed_42.pkl"
)

if ~RESULTS_DIR.exists():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def get_dir_path():
    for subfolder in DATA_DIR.iterdir():
        yield subfolder

def solve_for_x0(x0, net):
    return net(torch.tensor(x0, dtype=torch.float)).detach().numpy()

def main(dir):
    model = instanciate_model_object("cartpole")
    mpc = MPCBaseClass(model, N=20)
    if not (dir / f"performance_final_feasible_{FEASIBLE}_avg_{ALPHA_SPAN}.csv").exists():
        with open(dir / "args.yaml", "r") as f:
            args_dict = yaml.load(f, Loader=yaml.FullLoader)
        logger.info(f"{args_dict['exp_dir']=}")
        net = create_fc_net(
            input_dim=model.ns,
            output_dim=model.na,
            hidden_dims=args_dict["nn_depth"] * [args_dict["nn_width"]],
            tanh=TANH,
        )
        net.load_state_dict(torch.load(dir / "weights_nn_final.ckpt"))
        mpc.model.set_objective_ocp()

        logger.info("Evaluating policy for 2000 samples and N_rollout=50")
        policy = lambda x: solve_for_x0(x, net=net)
        with open(SAMPLES_DIR / SAMPLE_NAME, "rb") as f:
            x0_samples = pickle.load(f)
        logger.info("Samples for evaluation loaded")
        eval_stats_dict = evaluate_policy(
            policy, mpc.model, max_horizon=50, num_episodes=2000, samples=x0_samples
        )

        log_eval_stats(eval_stats_dict)
        save_eval_stats_to_csv(
            eval_stats_dict, dir, f"performance_final_feasible_{FEASIBLE}_avg_{ALPHA_SPAN}.csv"
        )
        save_full_eval_stats_to_csv(
            eval_stats_dict, dir, f"performance_final_feasible_{FEASIBLE}_full_{ALPHA_SPAN}.csv"
        )


def compute_stats_mpc_policy():

    if not (RESULTS_DIR / f"performance_avg_feasible_{FEASIBLE}_{ALPHA_SPAN}_mpc.csv").exists():
        model = instanciate_model_object(dynamic_model="cartpole")
        mpc = MPCBaseClass(model=model, N=20)

        with open(SAMPLES_DIR / SAMPLE_NAME, "rb") as f:
            x0_samples = pickle.load(f)
            print("Samples loaded")

        policy = lambda init_state: mpc.solve_for_x0(init_state)

        eval_stats_dict = evaluate_policy(policy, mpc.model, samples=x0_samples)
        save_eval_stats_to_csv(
            eval_stats_dict,
            exp_dir=RESULTS_DIR,
            filename=f"performance_avg_feasible_{FEASIBLE}_{ALPHA_SPAN}_mpc.csv",
        )
        save_full_eval_stats_to_csv(
            eval_stats_dict,
            exp_dir=RESULTS_DIR,
            filename=f"performance_full_feasible_{FEASIBLE}_{ALPHA_SPAN}_mpc.csv",
        )


def compute_average_stats():
    csv_name = f"performance_final_feasible_{FEASIBLE}_full_{ALPHA_SPAN}.csv"

    df = pd.DataFrame()
    for subfolder in DATA_DIR.iterdir():
        new_df = pd.read_csv(subfolder / csv_name)
        with open(subfolder / "args.yaml", "r") as f:
            args_dict = yaml.load(f, Loader=yaml.FullLoader)
        loss = args_dict["loss"]
        if loss == "vfn":
            loss = "q"
            qp_approx = args_dict["qp_approx"]
            if qp_approx:
                loss = "a-q"
        new_df["loss"] = loss
        new_df["cost"] = new_df["tracking_costs"] + new_df["violation_costs"]
        new_df = new_df.loc[
            new_df["cost"]
            < new_df["cost"].quantile(QUANTILE, interpolation="lower")
        ]
        new_df = new_df[["loss", "cost", "violation_bool"]].groupby("loss").mean()
        df = pd.concat([df, new_df.reset_index()], ignore_index=True)

    if (RESULTS_DIR / f"performance_full_feasible_{FEASIBLE}_{ALPHA_SPAN}_mpc.csv").exists():
        new_df = pd.read_csv(RESULTS_DIR / f"performance_full_feasible_{FEASIBLE}_{ALPHA_SPAN}_mpc.csv")
        new_df["loss"] = "mpc"
        new_df["cost"] = new_df["tracking_costs"] + new_df["violation_costs"]
        new_df = new_df.loc[
            new_df["cost"]
            < new_df["cost"].quantile(QUANTILE, interpolation="lower")
        ]
        new_df = new_df[["loss", "cost", "violation_bool"]].groupby("loss").mean()
        df = pd.concat([df, new_df.reset_index()], ignore_index=True)
    else:
        logger.info(
            "Stats for MPC evaluation not available yet, run compute_stats_mpc_policy() first!"
        )

    mean_df = df.groupby("loss").mean()
    std_df = df.groupby("loss").std()
    mean_df = mean_df.rename(
        columns={"cost": "cost_mean", "violation_bool": "violation_mean"}
    )
    std_df = std_df.rename(
        columns={"cost": "cost_std", "violation_bool": "violation_std"}
    )

    final_df = pd.merge(mean_df, std_df, left_index=True, right_index=True)
    final_df = final_df[["cost_mean", "cost_std", "violation_mean", "violation_std"]]
    final_df.to_csv(
        RESULTS_DIR
        / f"{DATA_DIR.name}_policy_evaluation_best_conf_feasible_{FEASIBLE}_span_{ALPHA_SPAN}_quant_{QUANTILE}.csv"
    )
    logger.info("Generated csv file with average stats")


if __name__ == "__main__":
    compute_stats_mpc_policy()
    process_map(main, get_dir_path(), max_workers=4)
    compute_average_stats()
