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

import logging
import pickle
import random
from argparse import ArgumentParser
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import yaml
from tqdm import tqdm

from imitate.autograd import QMpc
from imitate.mpc_acados import MPCBaseClass
from imitate.utils import (
    DaggerWrapper,
    collect_samples_dagger,
    create_fc_net,
    evaluate_policy,
    instanciate_model_object,
    log_eval_stats,
    save_eval_stats_to_csv,
    save_full_eval_stats_to_csv,
    set_seed,
)

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def create_argument_parser(**kw):
    args = ArgumentParser()
    args.add_argument(
        "--dynamic-model",
        type=str,
        default="cartpole",
        help="can be either 'cartpole' or 'inverted_pendulum'",
    )
    args.add_argument("--ocp-horizon", type=int, default=20)
    args.add_argument("--exp-dir", type=str)
    args.add_argument("--loss", type=str, default="vfn")
    args.add_argument("--collect-samples", type=str, default=True)
    args.add_argument("--dagger", type=int, default=20)
    args.add_argument("--eval-mpc", type=str, default=False)
    args.add_argument("--num-rollout", type=int, default=0)
    args.add_argument("--num-training-steps", type=int, default=2000)
    args.add_argument("--batch-size", type=int, default=32)
    args.add_argument("--log-level", type=str, default="WARNING")
    args.add_argument("--seed", type=int, default=0)
    args.add_argument("--nn-width", type=int, default=128)
    args.add_argument("--nn-depth", type=int, default=2)
    args.add_argument("--learning-rate", type=float, default=1e-3)
    args.add_argument("--save-net", type=str, default=True)
    args.add_argument("--tanh", type=int, default=1)
    args.add_argument("--qp-approx", type=str, default=False)
    args.add_argument("--save-checkpoint", type=str, default=False)
    args.add_argument("--no-cython", default=True, action="store_true")

    args = args.parse_args(**kw)

    return args


def main_train(args):
    # Create a new directory for the model and saving the performance, parameters and checkpoints.
    exp_dir = Path(args.exp_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)
    exp_dir.cwd()

    if (exp_dir / "stats.csv").exists():
        logger.info(f"aleady computed {exp_dir=}")

    else:
        # Save the arguments as a yaml file.
        with open(exp_dir / "args.yaml", "w") as f:
            yaml.dump(vars(args), f)

        logging.basicConfig(level=args.log_level)
        set_seed(args.seed)

        N = args.ocp_horizon
        model = instanciate_model_object(args.dynamic_model)
        mpc = MPCBaseClass(model=model, N=N, no_cython=args.no_cython, c_code_dir=exp_dir / "c_code_acados")
        dagger_env = None if args.dagger < 1 else DaggerWrapper(mpc)

        if args.eval_mpc:
            policy = lambda x: mpc.solve_ocp(x)["controls"][0]
            eval_stats_dict = evaluate_policy(policy, mpc.model, max_horizon=N)
            log_eval_stats(eval_stats_dict)

        if args.collect_samples and args.dagger < 1:
            logger.info("Collecting samples:")
            buffer = mpc.collect_initial_state(N_samples=1000, feasible=True)

            dcts = []
            for x0 in (pbar := tqdm(buffer, desc="Calculating optimal loss")):
                dcts.append(mpc.solve_ocp(x0[None, :]))

            optimal_obj = np.mean([dct["value_cost_fn"] for dct in dcts])
            optimal_u = np.array([dct["controls"][0, :] for dct in dcts], dtype=np.float32)
            traj_linearization = [
                {
                    "states": dct["states"],
                    "controls": dct["controls"],
                    "slacks_lower": dct["slacks"]["lower"],
                    "slacks_upper": dct["slacks"]["upper"],
                }
                for dct in dcts
            ]

            (exp_dir / "samples").mkdir(parents=True, exist_ok=True)
            with open(Path(exp_dir / "samples.pkl"), "wb") as f:
                pickle.dump((buffer, optimal_obj, optimal_u), f)
                logger.info("Samples saved")
        elif args.dagger > 0:
            buffer = deque()
            optimal_u = deque()
            optimal_obj = deque()
            traj_linearization = deque()
        else:
            logger.info("Loading samples:")
            with open(Path(exp_dir / "samples.pkl"), "rb") as f:
                buffer, optimal_obj, optimal_u, traj_linearization = pickle.load(f)

        logger.info("Training network:")
        net = create_fc_net(
            mpc.model.ns, mpc.model.na, [args.nn_width] * args.nn_depth, args.tanh
        )

        optimizer = optim.Adam(net.parameters(), lr=args.learning_rate)

        def loss_fn(x0_batch, u0_batch, u0_star, cost, lin_vec):
            losses = {}
            logger.debug(f"u0_star : {u0_star}")
            if args.loss == "vfn":
                losses["vfn"] = (
                    QMpc().apply(x0_batch, u0_batch, mpc, args.qp_approx, lin_vec)
                    + cost.mean()
                )
            elif args.loss == "l2":
                losses["vfn"] = torch.tensor([999.9])
            losses["l2"] = ((0.5 * (u0_batch - u0_star)) ** 2).mean()

            assert args.loss in losses
            return losses[args.loss], losses

        stats = defaultdict(list)

    # --------------------------------------- TRAINING LOOP ---------------------------------------

        for idx in (pbar := tqdm(range(args.num_training_steps))):

            if args.save_checkpoint:
                if idx%250==0:
                    logger.info(f"Evaluating policy after {idx} steps - on 2000 samples and N_rollout=50")
                    policy = lambda x: net(torch.tensor(x, dtype=torch.float)).detach().numpy()
                    with open("samples/cartpole_eval_samples_hp_sweep.pkl", "rb") as f:
                        x0_samples = pickle.load(f)
                    logger.info("Samples for evaluation loaded")
                    eval_stats_dict = evaluate_policy(policy, mpc.model, max_horizon=50, num_episodes=2000, samples=x0_samples)
                    logger.info(f"{args.exp_dir=}")
                    log_eval_stats(eval_stats_dict)
                    save_eval_stats_to_csv(eval_stats_dict, exp_dir, f"net_convergence_step_{idx}.csv")
                    if args.save_net:
                        torch.save(net.state_dict(), exp_dir / f"weights_nn_step_{idx}.ckpt")

            if idx % args.dagger == 0 and args.dagger > 0:
                policy = lambda x: net(torch.tensor(x, dtype=torch.float)).detach().numpy()
                # Collect samples
                fraction = 1 - (idx / args.num_training_steps)
                s, u, obj, traj = collect_samples_dagger(
                    dagger_env, policy, fraction_expert=fraction
                )
                buffer.extend(s)
                optimal_u.extend(u)
                optimal_obj.extend(obj)
                traj_linearization.extend(traj)
                assert (
                    len(buffer)
                    == len(optimal_u)
                    == len(optimal_obj)
                    == len(traj_linearization)
                )

            sample_idx = random.choices(range(len(buffer)), k=args.batch_size)
            x0_batch = torch.tensor(np.array([buffer[i] for i in sample_idx]), dtype=torch.float)
            u0_batch = net(x0_batch)
            u0_star = torch.tensor(np.array([optimal_u[i] for i in sample_idx]), dtype=torch.float)
            if args.qp_approx:
                w_lin_states = torch.tensor(np.array([traj_linearization[i]["states"] for i in sample_idx]), dtype=torch.float)
                w_lin_controls = torch.tensor(np.array([traj_linearization[i]["controls"] for i in sample_idx]), dtype=torch.float)
                w_lin_slacks_u = torch.tensor(np.array([traj_linearization[i]["slacks_upper"] for i in sample_idx]), dtype=torch.float)
                w_lin_slacks_l = torch.tensor(np.array([traj_linearization[i]["slacks_lower"] for i in sample_idx]), dtype=torch.float)
                w_lin = [w_lin_states, w_lin_controls, w_lin_slacks_l, w_lin_slacks_u]
            else:
                w_lin = None

            cost = torch.zeros(args.batch_size, dtype=torch.float)
            for _ in range(args.num_rollout):
                x0_batch, stage_cost = Shooting().apply(x0_batch, u0_batch, mpc.model)
                u0_batch = net(x0_batch)
                cost += stage_cost

            loss, stats_loss = loss_fn(x0_batch, u0_batch, u0_star, cost, lin_vec=w_lin)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            stats["vfn"].append(stats_loss["vfn"].item())
            stats["l2"].append(stats_loss["l2"].item())
            stats["umax"].append(u0_batch.max().item())
            stats["umin"].append(u0_batch.min().item())

            pbar.set_description(
                f"vfn: {np.mean(stats['vfn'][-100 :]):.2f} (opt: {np.mean(optimal_obj):.2f}), "
                f"l2: {np.mean(stats['l2'][-100 :]):.2f}, "
                f"umin: {min(stats['umin'][-100 :]):.2f}, "
                f"umax: {max(stats['umax'][-100 :]):.2f}"
            )

            if args.save_net:
                torch.save(net.state_dict(), exp_dir / "weights_nn_final.ckpt")

            # save the training stats to a csv file
            with open(exp_dir / "stats.csv", "w") as f:
                df = pd.DataFrame(stats)
                df.to_csv(f, index=False)

            if args.eval_mpc:
                logger.info("Evaluating learned policy")
                policy = lambda x: net(torch.tensor(x, dtype=torch.float)).detach().numpy()
                with open("samples/cartpole_eval_samples_feasible_True_constraint_span_0.3_seed_42.pkl", "rb") as f:
                    x0_samples = pickle.load(f)
                logger.info("Samples for evaluation loaded")
                eval_stats_dict = evaluate_policy(policy, mpc.model, max_horizon=50, num_episodes=2000, samples=x0_samples)

                logger.info(f"{args.exp_dir=}")
                log_eval_stats(eval_stats_dict)
                save_eval_stats_to_csv(eval_stats_dict, exp_dir, "performance_final_avg.csv")
                save_full_eval_stats_to_csv(eval_stats_dict, exp_dir, "performance_final_full.csv")

                return eval_stats_dict


if __name__ == "__main__":
    args = create_argument_parser()
    main_train(args)
