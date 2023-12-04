import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import torch
import yaml
import pickle
import logging

from imitate.mpc_acados import MPCBaseClass
from imitate.utils import policy_rollout
from imitate.main_training import (
    create_fc_net,
    instanciate_model_object,
)
from acados_template import latexify_plot

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


RESULTS_DIR = Path("results")
DATA_DIR = Path("experiments/cartpole/hp_sweep")
SAMPLES_DIR = Path("samples")

SEED = 1
QUANTILE = 1
ALPHA_SPAN = 0.3
FEASIBLE = False
TANH = True
SAMPLE_NAME = (
    f"cartpole_eval_samples_feasible_{FEASIBLE}_constraint_span_{ALPHA_SPAN}_seed_42.pkl"
)


def main():

    model = instanciate_model_object(dynamic_model="cartpole")
    mpc = MPCBaseClass(model=model, N=20)
    policies = {"l2-loss": {}, "q-loss": {}, "a-q-loss": {}, "v-loss": {}, "mpc": {}}

    with open(SAMPLES_DIR / SAMPLE_NAME, "rb") as f:
        x0_samples = pickle.load(f)
        print("Samples loaded")

    for dir in DATA_DIR.iterdir():
        with open(dir / "args.yaml", "r") as f:
            args_dict = yaml.load(f, Loader=yaml.FullLoader)
        if args_dict["seed"] == SEED:
            if args_dict["loss"] == "l2":
                policies["l2-loss"]["dir"] = dir
            elif args_dict["loss"] == "vfn":
                if args_dict["qp_approx"]:
                    policies["a-q-loss"]["dir"] = dir
                else:
                    policies["q-loss"]["dir"] = dir

    collect_u = {}
    collect_u["l2-loss"] = []
    collect_u["q-loss"] = []
    collect_u["a-q-loss"] = []

    for p in policies:
        x_list = []
        u_list = []

        if p == "mpc":
            policy = lambda init_state: mpc.solve_ocp(init_state)["controls"][0, :]
        else:
            breakpoint()
            try:
                with open(policies[p]["dir"] / "args.yaml", "r") as f:
                    args_dict = yaml.load(f, Loader=yaml.FullLoader)
                net = create_fc_net(
                    input_dim=model.ns,
                    output_dim=model.na,
                    hidden_dims=args_dict["nn_depth"] * [args_dict["nn_width"]],
                    tanh=TANH,
                )
                net.load_state_dict(
                    torch.load(policies[p]["dir"] / "weights_nn_final.ckpt")
                )
                policy = (
                    lambda init_state: net(torch.tensor(init_state, dtype=torch.float))
                    .detach()
                    .numpy()
                )
            except:
                policy = lambda x: np.zeros(1)

        # rng = np.random.default_rng(42)
        # for _ in range(100):
        for i, x0 in enumerate(x0_samples[:100, :]):
            # x0 = rng.uniform(low=CNS_SCALING * model.s_min, high=CNS_SCALING * model.s_max)
            # x0 = np.array([0, 0, np.pi/4, 0])

            # if (p == 'mpc') & (i == 2): # Early termination for MPC, it takes a lot otherwise!
            #     logger.info(f"Print only {i} MPC trajectory for quick plot!")
            #     break

            x, u, _, _, violations = policy_rollout(policy, model, x0, max_horizon=50)
            x_list += [x]
            u_list += [u]
        policies[p]["u_list"] = u_list
        policies[p]["x_list"] = x_list


    latexify_plot()
    fig, axs = plt.subplots(4, 1, sharex=True, figsize=(4.6, 4.6))
    t = np.arange(0,50*0.05, 0.05)
    for u in policies['l2-loss']['u_list']:
        axs[0].step(t, u * model.scaling_factor_action, where='post', linewidth=1, c='tab:blue', alpha=0.4, label=r"$\mathcal{L}^2$")
    for u in policies['q-loss']['u_list']:
        axs[1].step(t, u * model.scaling_factor_action, where='post', linewidth=1, c='tab:blue', alpha=0.4, label=r"$\mathcal{L}^\mathrm{Q}$")
    for u in policies['a-q-loss']['u_list']:
        axs[2].step(t, u * model.scaling_factor_action, where='post', linewidth=1, c='tab:blue', alpha=0.4, label=r"$\mathcal{L}^\mathrm{Q_a}$")
    for u in policies['mpc']['u_list']:
        axs[3].step(t, u * model.scaling_factor_action, where='post', linewidth=1, c='tab:blue', alpha=0.4, label=r"$\mathcal{L}^\mathrm{MPC}$")

    axs[0].grid()
    han, l = axs[0].get_legend_handles_labels()

    axs[0].annotate(r"$\mathcal{L}^2$", xy=(2.0, 8))
    axs[0].set_ylim(1.3 * model.a_min * model.scaling_factor_action, 1.3 * model.a_max * model.scaling_factor_action)
    axs[0].axhline(model.scaling_factor_action * model.a_max, linestyle=":", color="k", alpha=0.5)
    axs[0].axhline(model.scaling_factor_action * model.a_min, linestyle=":", color="k", alpha=0.5)
    axs[0].set_ylabel(r"$u \; \mathrm{[rad/s^2]}$")

    axs[1].grid()
    axs[1].annotate(r"$\mathcal{L}^\mathrm{Q}$", xy=(2.0, 8))
    axs[1].set_ylim(1.3 * model.a_min * model.scaling_factor_action, 1.3 * model.a_max * model.scaling_factor_action)
    axs[1].axhline(model.scaling_factor_action * model.a_max, linestyle=":", color="k", alpha=0.5)
    axs[1].axhline(model.scaling_factor_action * model.a_min, linestyle=":", color="k", alpha=0.5)
    axs[1].set_ylabel(r"$u \; \mathrm{[rad/s^2]}$")

    axs[2].grid()
    axs[2].annotate(r"$\mathcal{L}^\mathrm{Q_a}$", xy=(2.0, 8))
    axs[2].set_ylim(1.3 * model.a_min * model.scaling_factor_action, 1.3 * model.a_max * model.scaling_factor_action)
    axs[2].axhline(model.scaling_factor_action * model.a_max, linestyle=":", color="k", alpha=0.5)
    axs[2].axhline(model.scaling_factor_action * model.a_min, linestyle=":", color="k", alpha=0.5)
    axs[2].set_ylabel(r"$u \; \mathrm{[rad/s^2]}$")

    axs[-1].grid()
    axs[-1].annotate(r"$\pi^\star$ MPC", xy=(2.0, 8))
    axs[-1].set_ylim(1.3 * model.a_min * model.scaling_factor_action, 1.3 * model.a_max * model.scaling_factor_action)
    axs[-1].axhline(model.scaling_factor_action * model.a_max, linestyle=":", color="k", alpha=0.5)
    axs[-1].axhline(model.scaling_factor_action * model.a_min, linestyle=":", color="k", alpha=0.5)
    axs[-1].set_ylabel(r"$u \; \mathrm{[rad/s^2]}$")
    axs[-1].set_xlabel(r"time [s]")
    axs[-1].set_xlim(0, )

    Path("figures").mkdir(parents=True, exist_ok=True)
    fig.savefig(f"figures/{DATA_DIR.name}_representative_rollouts_seed_{SEED}.pdf", bbox_inches="tight")

    plt.show()


if __name__ == "__main__":
    main()
