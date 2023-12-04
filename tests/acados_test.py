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
