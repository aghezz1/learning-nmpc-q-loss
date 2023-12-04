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
from imitate.mpc_acados import MPCBaseClass
from imitate.utils import instanciate_model_object, set_seed
import pickle
import os

SEED = 42
N_ocp_horizon = 20
MODEL = "cartpole"
FEASIBLE = True
SPAN = 0.3

samples_dir = Path("samples")
samples_dir.mkdir(parents=True, exist_ok=True)

name_samples = (
    f"{MODEL}_eval_samples_feasible_{FEASIBLE}_constraint_span_{SPAN}_seed_{SEED}.pkl"
)
set_seed(SEED)

model = instanciate_model_object(dynamic_model=MODEL)
mpc = MPCBaseClass(model=model, N=N_ocp_horizon)


x0_samples = mpc.collect_initial_state(
    N_samples=2000, feasible=FEASIBLE, constraint_span=SPAN
)
with open(samples_dir / name_samples, "wb") as f:
    pickle.dump(x0_samples, f)
    print("Samples saved")

with open(samples_dir / name_samples, "rb") as f:
    x0_samples = pickle.load(f)
    print("Samples loaded")
