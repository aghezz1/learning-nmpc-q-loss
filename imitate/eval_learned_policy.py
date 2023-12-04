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

from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
import logging

from imitate.models import InvertedPendulumModel
from imitate.mpc import MPCBaseClass
from imitate.utils import plot_trajectory, evaluate_policy
from imitate.main_training import (
    create_fc_net,
    instanciate_model_object,
)

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def main(args):
    N = 50
    exp_dir = Path(args.exp_dir)

    # Load yaml file
    with open(exp_dir / "args.yaml", "r") as f:
        args_dict = yaml.load(f, Loader=yaml.FullLoader)
    # Create model
    model = instanciate_model_object(dynamic_model=args_dict["dynamic_model"])
    model.set_constraints_ocp()
    model.set_objective_ocp()
    model.create_auxiliary_functions()
    # Load model
    net = create_fc_net(
        input_dim=model.ns,
        output_dim=model.na,
        hidden_dims=args_dict["nn_depth"] * [args_dict["nn_width"]],
    )
    net.load_state_dict(torch.load(exp_dir / "weights_nn.ckpt"))

    # ------------ EVALUATION FOR MULTIPLE INITIAL STATES --------------
    eval_stats_dict = evaluate_policy(
        lambda x: net(torch.tensor(x, dtype=torch.float)).detach().numpy(),
        model,
        num_episodes=1000,
        max_horizon=150,
        seed=42,
    )
    logger.info(
        f"Mean tracking cost: {eval_stats_dict['tracking_costs'].mean():.3f} +- {eval_stats_dict['tracking_costs'].std():.3f}"
    )
    logger.info(
        f"Mean violation cost: {eval_stats_dict['violation_costs'].mean():.3f} +- {eval_stats_dict['violation_costs'].std():.3f}"
    )
    logger.info(f"Violations: {eval_stats_dict['violations']:.3f}")

    # ------------ EVALUATION FOR A GIVEN INITIAL STATE --------------
    # Define initial state for plots
    if model.name == "cartpole":
        # x0_start = np.array([-0.1, 0.0, np.pi / 9, -0.1])
        # x0_start = np.array([0.0, 0.0, 0.0, 0.0])
        rng = np.random.default_rng(42)
        x0_start = rng.uniform(low=0.3 * model.s_min, high=0.3 * model.s_max)
    elif model.name == "inverted_pendulum":
        x0_start = np.array([np.pi / 6, -0.1])

    # Evaluation of learned policy
    model = instanciate_model_object(dynamic_model=args_dict["dynamic_model"])
    step_fn = model.integrator
    states = []
    controls = []
    x0 = torch.tensor(x0_start)
    states.append(np.array(x0))
    for _ in range(N):
        u0 = net(torch.tensor(x0, dtype=torch.float))
        # print(u0)
        # u0 = torch.clip(u0, -10.0, 10.0)
        controls.append(u0.detach().numpy())
        x0 = step_fn(np.array(x0), u0.detach().numpy()).full().squeeze()
        states.append(x0)
    states = np.vstack(states)
    controls = np.vstack(controls)

    # Evaluation of MPC policy
    mpc = MPCBaseClass(model=model, N=N)
    results_ocp_dict = mpc.solve_ocp(x0=x0_start)
    states_mpc = results_ocp_dict["states"]
    controls_mpc = results_ocp_dict["controls"]

    # Plot the policies
    plot_trajectory(
        states, controls, model, title="learned " + args_dict["loss"] + " policy"
    )
    plt.savefig(exp_dir / "eval_policy.png", dpi=300)
    plot_trajectory(states_mpc, controls_mpc, model, title="mpc policy")
    plt.savefig(exp_dir / "eval_mpc.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    args = ArgumentParser()
    args.add_argument("--exp-dir", type=str)
    args.add_argument("--plot-name", type=str, default=None)
    args = args.parse_args()
    main(args)
