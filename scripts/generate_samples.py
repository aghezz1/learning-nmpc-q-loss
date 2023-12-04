import numpy as np
from imitate.models import CartpoleModel
from imitate.mpc_acados import MPCBaseClass
from imitate.utils import instanciate_model_object, set_seed
import matplotlib.pyplot as plt
import pickle
import os

SEED = 42
N_ocp_horizon = 20
MODEL = "cartpole"
FEASIBLE = False
SPAN = 0.3

name_samples = (
    f"{MODEL}_eval_samples_feasible_{FEASIBLE}_constraint_span_{SPAN}_seed_{SEED}.pkl"
)
set_seed(SEED)

model = instanciate_model_object(dynamic_model=MODEL)
mpc = MPCBaseClass(model=model, N=N_ocp_horizon)


x0_samples = mpc.collect_initial_state(
    N_samples=2000, feasible=FEASIBLE, constraint_span=SPAN
)
with open(os.path.join("samples", name_samples), "wb") as f:
    pickle.dump(x0_samples, f)
    print("Samples saved")

with open(os.path.join("samples", name_samples), "rb") as f:
    x0_samples = pickle.load(f)
    print("Samples loaded")
