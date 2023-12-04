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

import numpy as np
from imitate.models import CartpoleModel
from imitate.mpc_acados import MPCBaseClass
import matplotlib.pyplot as plt
from acados_template import latexify_plot
from pathlib import Path

latexify_plot()

model = CartpoleModel(integrator="rk4")
mpc = MPCBaseClass(model=model, N=20, no_cython=True)
x0 = np.array([1, 0, np.pi / 4, 0])


ocp_result = mpc.solve_ocp(x0)
u0_star = ocp_result["controls"][0, :]
print(f"u0_star = {u0_star}")
print(f"cost = {ocp_result['value_cost_fn']}")

param_ocp_result = mpc.solve_parametric_ocp(x0, u0_star)
print(f"u1_star = {param_ocp_result['controls'][1,:]}")
print(f"cost = {param_ocp_result['value_cost_fn']}")

# %% NLP testing
# Keep x0 fixed, try different u0
possible_u = np.linspace(model.a_min * 1.5, model.a_max * 1.5, 49)
status_list, u1_star_list, objective_list, lam_param = [], [], [], []

for u0 in possible_u:
    print(f"{u0=}")
    param_ocp_result = mpc.solve_parametric_ocp(x0, u0)
    status_list.append(param_ocp_result["solver_stats"]["return_status"])
    if param_ocp_result["solver_stats"]["success"]:
        objective_list.append(param_ocp_result["value_cost_fn"])
        u1_star_list.append(param_ocp_result["controls"][1, :])
        lam_param.append(param_ocp_result["lambda0"])
    else:
        mpc.parametric_ocp_solver.print_statistics()
        # break
        objective_list.append(np.nan)
        u1_star_list.append(np.nan)
        lam_param.append(np.nan * np.ones(model.ns + model.na))

status_array = np.array(status_list)
failing = np.nonzero(status_array)[0]
print(f"{len(failing)} failing_runs: {failing}")

# %% QP approx testing
status_list_qp, u1_star_list_qp, objective_list_qp, lam_param_qp = [], [], [], []

for u0 in possible_u:
    param_ocp_result = mpc.solve_parametric_quadratic_ocp(x0, u0, lin_vector=ocp_result)
    status_list_qp.append(param_ocp_result["solver_stats"]["return_status"])
    if param_ocp_result["solver_stats"]["success"]:
        objective_list_qp.append(param_ocp_result["value_cost_fn"])
        u1_star_list_qp.append(param_ocp_result["controls"][0, :])
        lam_param_qp.append(param_ocp_result["lambda0"])
        # lam_slack_u0.append(param_ocp_result["sol_dict_raw"]["lam_x"][:2].full().squeeze())
    else:
        objective_list_qp.append(np.nan)
        u1_star_list_qp.append(np.nan)
        lam_param_qp.append(np.nan * np.ones(2 * model.na))
        # lam_slack_u0.append(np.nan * np.ones(model.na))

lam_param_qp = np.array(lam_param_qp)
status_array = np.array(status_list_qp)
failing = np.nonzero(status_array)
print(f"{len(failing)} failing_runs: {failing}")

# %% Plotting
x = np.linspace(-1.5, 1.5, 200)
y = lambda x, y_bar, x_bar, dydu: y_bar + dydu * (x - x_bar)

fig, ax = plt.subplots(1, 1, figsize=(7, 7))
ax.plot(possible_u, objective_list, linewidth=3, label=r"parametric NLP")
for idx in range(0, len(possible_u), 5):
    ax.plot(
        possible_u,
        y(possible_u, objective_list[idx], possible_u[idx], lam_param[idx]),
        alpha=0.2,
        c="pink",
    )

ax.plot(
    possible_u,
    objective_list_qp,
    linewidth=3,
    label=r"parametric QP($x_0^\star, u_0^\star$)",
)
for idx in range(0, len(possible_u), 5):
    ax.plot(
        possible_u,
        y(possible_u, objective_list_qp[idx], possible_u[idx], lam_param_qp[idx]),
        alpha=0.2,
        c="black",
    )

ax.plot(
    possible_u,
    model.scaling_factor_action * (possible_u - u0_star) ** 2,
    "--",
    label=r"$L_2$ loss",
)

ax.scatter(u0_star, ocp_result["value_cost_fn"], marker="*", c="r", s=50, alpha=1)

plt.legend()
plt.xlim(min(possible_u), max(possible_u))
plt.ylim(
    0,
)
plt.xlabel("$u_0$")
plt.ylabel("$J(x_0, u_0)$")
plt.tight_layout()
figure_dir = Path("figures")
figure_dir.mkdir(parents=True, exist_ok=True)
figure_dir.cwd()
plt.savefig(figure_dir / "acados_test_loop.png", dpi=200)
plt.show()
