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
from acados_template import latexify_plot

latexify_plot()

model = CartpoleModel(integrator="rk4")
mpc = MPCBaseClass(model=model, N=20)
x0 = np.array([0.2, 0, np.pi / 4, 0])


ocp_result = mpc.solve_ocp(x0)
u0_star = ocp_result["controls"][0, :]
# print(f"u0_star = {u0_star}")
print(f"cost = {ocp_result['value_cost_fn']}")
# print(f"{ocp_result['controls']=}")
# print(f"{ocp_result['states']=}")
# print(f"{ocp_result=}")

par_ocp_results = mpc.solve_parametric_ocp(x0, u0_star)
print(f"cost = {par_ocp_results['value_cost_fn']}")
# print(f"{par_ocp_results['controls']=}")
# print(f"{par_ocp_results['states']=}")
