import copy
import logging
import tempfile
from pathlib import Path
from typing import Optional

import casadi as ca
import numpy as np
import scipy.linalg as LA
from acados_template import AcadosOcp, AcadosOcpSolver

from imitate.models import MpcModelBase

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

INF = 1e8


class MPCBaseClass:
    def __init__(
        self,
        model: MpcModelBase,
        N: int,
        no_cython: bool = True,
        c_code_dir: Optional[Path] = None,
    ):
        self.model = model
        self.N = N  # horizon length
        self.no_cython = no_cython

        self.model.set_constraints_ocp(s_min=None, s_max=None, a_min=None, a_max=None)
        self.model.set_objective_ocp(Q=None, R=None)
        self.model.create_auxiliary_functions()
        if model.name == "cartpole":
            self.weight_l1_x = np.array([10, 1, 10, 1])
            self.weight_l2_x = 2 * np.array([1e3, 1e2, 1e3, 1e2])
            self.weight_l1_u = np.array([1e5])
            self.weight_l2_u = 2 * np.array([1e4])
        elif model.name == "inverted_pendulum":
            self.weight_l1_x = np.array([10, 1])
            self.weight_l1_u = np.array([1e4])
        self.standard_solver_created = False
        self.parametric_solver_created = False
        self.qp_solver_created = False
        self.prob_dict = None
        self.backward_solver = None
        self.ocp_bound_dict = None
        self.parametric_ocp_bound_dict = None
        self.solver_options = {
            "tol": 1e-5,
            # "qp_tol": 1e-6,
            "nlp_solver_type": "SQP",
            "qp_solver": "PARTIAL_CONDENSING_HPIPM",
            "qp_solver_cond_N": self.N,
            # "qp_solver_warm_start": 2,
            "hessian_approx": "EXACT",
            "integrator_type": "ERK",
            "nlp_solver_max_iter": 250,
            "qp_solver_iter_max": 400,
            "levenberg_marquardt": 0.0,
        }
        self.qp_solver_options = copy.deepcopy(self.solver_options)
        self.qp_solver_options["nlp_solver_type"] = "SQP_RTI"
        self.qp_solver_options["hessian_approx"] = "GAUSS_NEWTON"
        self.c_code_dir = (
            c_code_dir
            if c_code_dir is not None
            else Path(__file__).parent.parent.parent / "c_code_acados"
        )
        self.c_code_dir = self.c_code_dir.resolve()
        self.c_code_dir.mkdir(parents=True, exist_ok=True)

    def _build_ocp(self) -> tuple:
        """parameters: x0 -> solve for u0"""
        N = self.N
        # Formulate the ocp-nlp
        ocp = AcadosOcp()
        tmp_dir = tempfile.mkdtemp(prefix="acados_ocp_", dir=self.c_code_dir)
        ocp.code_export_directory = tmp_dir
        ocp.model.x = self.model.s
        ocp.model.u = self.model.a
        ocp.model.f_expl_expr = self.model.s_dot
        ocp.model.name = self.model.name  # + "_" + str(time.time_ns())

        nx = self.model.ns
        nu = self.model.na
        ny = nx + nu
        ocp.dims.N = N
        # cost
        ocp.cost.cost_type = "NONLINEAR_LS"
        ocp.cost.cost_type_e = "NONLINEAR_LS"
        ocp.model.cost_y_expr = ca.vertcat(self.model.s, self.model.a)
        ocp.model.cost_y_expr_e = self.model.s
        ocp.cost.yref_e = np.zeros(nx)
        ocp.cost.yref = np.zeros(ny)
        ocp.cost.W = LA.block_diag(self.model.Q, self.model.R) / N
        ocp.cost.W_e = self.model.lin_model["P_N"] / N
        # setting solver options
        ocp.solver_options.tf = self.model.dt * N
        ocp.solver_options.nlp_solver_type = self.solver_options["nlp_solver_type"]
        ocp.solver_options.qp_solver = self.solver_options["qp_solver"]
        ocp.solver_options.qp_solver_iter_max = self.solver_options[
            "qp_solver_iter_max"
        ]
        # ocp.solver_options.levenberg_marquardt = self.solver_options["levenberg_marquardt"]
        ocp.solver_options.hessian_approx = self.solver_options["hessian_approx"]
        ocp.solver_options.integrator_type = self.solver_options["integrator_type"]
        ocp.solver_options.nlp_solver_type = self.solver_options["nlp_solver_type"]
        ocp.solver_options.nlp_solver_max_iter = self.solver_options[
            "nlp_solver_max_iter"
        ]
        ocp.solver_options.tol = self.solver_options["tol"]
        # constraints
        ocp.constraints.idxbx = np.arange(self.model.ns)
        ocp.constraints.lbx = self.model.s_min
        ocp.constraints.ubx = self.model.s_max
        ocp.constraints.idxbx_e = np.arange(self.model.ns)
        ocp.constraints.lbx_e = self.model.s_min
        ocp.constraints.ubx_e = self.model.s_max
        ocp.constraints.lbu = self.model.a_min
        ocp.constraints.ubu = self.model.a_max
        ocp.constraints.idxbu = np.arange(self.model.na)
        ocp.constraints.x0 = np.zeros(self.model.ns)
        # slacks
        ocp.constraints.idxsbx = np.arange(self.model.ns)
        ocp.constraints.idxsbx_e = np.arange(self.model.ns)
        ocp.cost.Zl = self.weight_l2_x / N
        ocp.cost.zl = self.weight_l1_x / N
        ocp.cost.Zl_e = self.model.dt * self.weight_l2_x / N
        ocp.cost.zl_e = self.model.dt * self.weight_l1_x / N
        ocp.cost.Zu = self.weight_l2_x / N
        ocp.cost.zu = self.weight_l1_x / N
        ocp.cost.Zu_e = self.model.dt * self.weight_l2_x / N
        ocp.cost.zu_e = self.model.dt * self.weight_l1_x / N

        if self.no_cython:
            self.ocp_solver = AcadosOcpSolver(
                ocp, json_file=tmp_dir + "/acados_ocp_nlp.json"
            )
        else:
            solver_json = tmp_dir + "/acados_ocp_nlp.json"
            AcadosOcpSolver.generate(ocp, json_file=solver_json)
            AcadosOcpSolver.build(ocp.code_export_directory, with_cython=True)
            self.ocp_solver = AcadosOcpSolver.create_cython_solver(solver_json)
        # self.ocp_solver.store_iterate(filename="acados_def_iter/ocp_def_iter.json", overwrite=True)
        self.standard_solver_created = True

    def _build_parametric_ocp(self) -> tuple:
        """parameters: x0, u0 -> solve for u1, lam_u0"""
        N = self.N
        # Formulate the ocp-nlp
        parametric_ocp = AcadosOcp()
        tmp_dir = tempfile.mkdtemp(prefix="acados_param_ocp_", dir=self.c_code_dir)
        parametric_ocp.code_export_directory = tmp_dir
        parametric_ocp.model.x = self.model.s
        parametric_ocp.model.u = self.model.a
        parametric_ocp.model.f_expl_expr = self.model.s_dot
        parametric_ocp.model.name = (
            self.model.name
        )  # + "_parametric" + "_" + str(time.time_ns())

        nx = self.model.ns
        nu = self.model.na
        ny = nx + nu
        parametric_ocp.dims.N = N
        # cost
        parametric_ocp.cost.cost_type = "NONLINEAR_LS"
        parametric_ocp.cost.cost_type_e = "NONLINEAR_LS"
        parametric_ocp.model.cost_y_expr = ca.vertcat(self.model.s, self.model.a)
        parametric_ocp.model.cost_y_expr_e = self.model.s
        parametric_ocp.cost.yref = np.zeros(ny)
        parametric_ocp.cost.yref_e = np.zeros(nx)
        parametric_ocp.cost.W = LA.block_diag(self.model.Q, self.model.R) / N
        parametric_ocp.cost.W_e = self.model.lin_model["P_N"] / N
        # setting solver options
        parametric_ocp.solver_options.tf = self.model.dt * N
        parametric_ocp.solver_options.nlp_solver_type = self.solver_options[
            "nlp_solver_type"
        ]
        parametric_ocp.solver_options.qp_solver = self.solver_options["qp_solver"]
        parametric_ocp.solver_options.qp_solver_iter_max = self.solver_options[
            "qp_solver_iter_max"
        ]
        # parametric_ocp.solver_options.levenberg_marquardt = self.solver_options["levenberg_marquardt"]
        parametric_ocp.solver_options.hessian_approx = self.solver_options[
            "hessian_approx"
        ]
        parametric_ocp.solver_options.integrator_type = self.solver_options[
            "integrator_type"
        ]
        parametric_ocp.solver_options.nlp_solver_type = self.solver_options[
            "nlp_solver_type"
        ]
        parametric_ocp.solver_options.nlp_solver_max_iter = self.solver_options[
            "nlp_solver_max_iter"
        ]
        parametric_ocp.solver_options.tol = self.solver_options["tol"]
        parametric_ocp.solver_options.nlp_solver_tol_stat = (
            self.solver_options["tol"] * 1e1
        )
        parametric_ocp.solver_options.qp_solver_tol_eq = (
            self.solver_options["tol"] * 1e-2
        )
        parametric_ocp.solver_options.qp_solver_tol_ineq = (
            self.solver_options["tol"] * 1e-1
        )
        parametric_ocp.solver_options.qp_solver_tol_stat = (
            self.solver_options["tol"] * 1e-1
        )
        parametric_ocp.solver_options.qp_solver_tol_comp = (
            self.solver_options["tol"] * 1e-1
        )
        parametric_ocp.solver_options.nlp_solver_ext_qp_res = 0
        # parametric_ocp.solver_options.globalization = 'MERIT_BACKTRACKING'
        # parametric_ocp.solver_options.globalization_use_SOC = 1
        # parametric_ocp.solver_options.alpha_min = 0.2
        parametric_ocp.solver_options.regularize_method = "PROJECT"

        # constraints
        parametric_ocp.constraints.x0 = np.zeros(self.model.ns)
        parametric_ocp.constraints.idxbu = np.arange(self.model.na)
        parametric_ocp.constraints.lbu = -INF * np.ones([self.model.na])
        parametric_ocp.constraints.ubu = INF * np.ones([self.model.na])

        # g constraint needed instead of bound on u because we need to slack u_0
        parametric_ocp.constraints.C = np.vstack(
            [np.diag(np.ones(self.model.ns)), np.zeros((self.model.na, self.model.ns))]
        )
        parametric_ocp.constraints.D = np.vstack(
            [np.zeros((self.model.ns, self.model.na)), np.diag(np.ones(self.model.na))]
        )
        parametric_ocp.constraints.lg = np.concatenate(
            [self.model.s_min, self.model.a_min]
        )
        parametric_ocp.constraints.ug = np.concatenate(
            [self.model.s_max, self.model.a_max]
        )
        parametric_ocp.constraints.C_e = np.diag(np.ones(self.model.ns))
        parametric_ocp.constraints.lg_e = self.model.s_min
        parametric_ocp.constraints.ug_e = self.model.s_max
        # slacks
        parametric_ocp.constraints.idxsg = np.arange(self.model.ns + self.model.na)
        parametric_ocp.constraints.idxsg_e = np.arange(self.model.ns)
        parametric_ocp.cost.Zl = (
            np.concatenate([self.weight_l2_x, self.weight_l2_u]) / N
        )
        parametric_ocp.cost.zl = (
            np.concatenate([self.weight_l1_x, self.weight_l1_u]) / N
        )
        parametric_ocp.cost.Zl_e = self.model.dt * self.weight_l2_x / N
        parametric_ocp.cost.zl_e = self.model.dt * self.weight_l1_x / N
        parametric_ocp.cost.Zu = (
            np.concatenate([self.weight_l2_x, self.weight_l2_u]) / N
        )
        parametric_ocp.cost.zu = (
            np.concatenate([self.weight_l1_x, self.weight_l1_u]) / N
        )
        parametric_ocp.cost.Zu_e = self.model.dt * self.weight_l2_x / N
        parametric_ocp.cost.zu_e = self.model.dt * self.weight_l1_x / N

        if self.no_cython:
            self.parametric_ocp_solver = AcadosOcpSolver(
                parametric_ocp, json_file=tmp_dir + "/acados_ocp_nlp.json"
            )
        else:
            solver_json = tmp_dir + "/acados_ocp_nlp.json"
            AcadosOcpSolver.generate(parametric_ocp, json_file=solver_json)
            AcadosOcpSolver.build(
                parametric_ocp.code_export_directory, with_cython=True
            )
            self.parametric_ocp_solver = AcadosOcpSolver.create_cython_solver(
                solver_json
            )
        # self.parametric_ocp_solver.store_iterate(filename="acados_def_iter/param_ocp_def_iter.json", overwrite=True)
        self.parametric_solver_created = True

    def solve_ocp(self, x0: np.ndarray) -> dict:
        if not self.standard_solver_created:
            self._build_ocp()

        if len(x0.shape) == 2:
            x0 = x0.squeeze()
        states = np.zeros((self.N + 1, self.model.ns))
        slacks0 = np.zeros((self.N + 1, self.model.ns))
        slacks = {"lower": slacks0, "upper": slacks0}
        controls = np.zeros((self.N, self.model.na))

        # initialize solver
        self.ocp_solver.load_iterate("acados_def_iter/ocp_def_iter.json")
        self.ocp_solver.set(0, "x", x0)
        self.ocp_solver.set(0, "lbx", x0)
        self.ocp_solver.set(0, "ubx", x0)

        # solve
        solver_stats = {}
        solver_stats["return_status"] = self.ocp_solver.solve()
        value_cost_fn = self.ocp_solver.get_cost()

        # logger.info(f"Time tot [sec]: {self.ocp_solver.get_stats('time_tot')}, sqp_iter: {self.ocp_solver.get_stats('sqp_iter')}")
        if solver_stats["return_status"] != 0:
            # self.ocp_solver.print_statistics()
            logger.warning(
                f"acados solver returned status = {solver_stats['return_status']}"
            )
            solver_stats["success"] = 0
        else:
            solver_stats["success"] = 1

        for i in range(self.N + 1):
            states[i, :] = self.ocp_solver.get(i, "x")
        for i in range(1, self.N):
            slacks["lower"][i, :] = self.ocp_solver.get(i, "sl")
            slacks["upper"][i, :] = self.ocp_solver.get(i, "su")
        for i in range(self.N):
            controls[i, :] = self.ocp_solver.get(i, "u")

        # order: lbu, lbx, ubu, ubx
        lambda0 = self.ocp_solver.get(0, "lam")

        return {
            "value_cost_fn": value_cost_fn,
            "states": states,
            "controls": controls,
            "slacks": slacks,
            "solver_stats": solver_stats,
            "lambda0": lambda0,
        }

    def solve_parametric_ocp(self, x0: np.ndarray, u0: np.ndarray) -> dict:
        if not self.parametric_solver_created:
            self._build_parametric_ocp()

        states = np.zeros((self.N + 1, self.model.ns))
        controls = np.zeros((self.N, self.model.na))

        # set initial state
        # initialize solver? # TODO
        # self.parametric_ocp_solver.load_iterate("default_iterate_parametric" + self.parametric_ocp_solver.model_name + ".json")
        self.parametric_ocp_solver.set(0, "u", u0)
        for i in range(self.N + 1):  # setting xk from x0=x0 to xN=0
            self.parametric_ocp_solver.set(i, "x", (1 - i / self.N + 1) * x0)

        self.parametric_ocp_solver.set(0, "lbx", x0)
        self.parametric_ocp_solver.set(0, "ubx", x0)
        self.parametric_ocp_solver.set(0, "lbu", u0)
        self.parametric_ocp_solver.set(0, "ubu", u0)

        # solve
        solver_stats = {}
        solver_stats["return_status"] = self.parametric_ocp_solver.solve()
        value_cost_fn = self.parametric_ocp_solver.get_cost()

        logger.info(
            f"Time tot [sec]: {self.parametric_ocp_solver.get_stats('time_tot')}, sqp_iter: {self.parametric_ocp_solver.get_stats('sqp_iter')}"
        )
        if solver_stats["return_status"] != 0:
            # self.parametric_ocp_solver.print_statistics()
            logger.warning(
                f"acados solver returned status = {solver_stats['return_status']}"
            )
            solver_stats["success"] = 0
        else:
            solver_stats["success"] = 1

        for i in range(self.N + 1):
            states[i, :] = self.parametric_ocp_solver.get(i, "x")
        for i in range(self.N):
            controls[i, :] = self.parametric_ocp_solver.get(i, "u")

        # order: [ lbu lbx lg  ubu ubx ug  lsg usg]
        ny = self.model.na + self.model.ns
        lambda0 = self.parametric_ocp_solver.get(0, "lam")[: self.model.na]  # lbu
        lambda0 += -self.parametric_ocp_solver.get(0, "lam")[
            2 * ny : 2 * ny + self.model.na
        ]  # ubu
        # logger.info(f"{lambda0=},")

        return {
            "value_cost_fn": value_cost_fn,
            "states": states,
            "controls": controls,
            "solver_stats": solver_stats,
            "lambda0": lambda0,
        }

    def collect_initial_state(
        self, N_samples: int, feasible: bool = False, constraint_span: float = 0.2
    ) -> np.ndarray:
        if self.model.name == "inverted_pendulum":
            if feasible:
                logger.info("Collecting feasible samples")
                raise NotImplementedError("not implemented for inverted pendulum yet")
            else:
                logger.info("Collecting generic samples in the specified bounds")
                low = [-80 * np.pi / 180, -1]
                high = [90 * np.pi / 180, 1]
                return np.random.uniform(low, high, size=(N_samples, self.model.ns))

        elif self.model.name == "cartpole":
            low = constraint_span * self.model.s_min
            high = constraint_span * self.model.s_max
            if not feasible:
                logger.info("Collecting generic samples in the specified bounds")
                return np.random.uniform(low, high, size=(N_samples, self.model.ns))
            else:
                logger.info("Collecting feasible samples")
                samples = []
                success = []
                done = False
                while not done:
                    x0 = np.random.uniform(low, high, size=(self.model.ns,))
                    samples.append(x0)
                    ocp_result = self.solve_ocp(x0)
                    if np.any(
                        np.abs(ocp_result["states"]).max(axis=0)
                        > np.abs(1.1 * self.model.s_max)
                    ):
                        success.append(0)
                    elif np.any(
                        np.abs(ocp_result["controls"]).max(axis=0)
                        > np.abs(1.1 * self.model.a_max)
                    ):
                        success.append(0)
                    else:
                        success.append(1)
                    if sum(success) >= N_samples:
                        done = True
                samples = np.array(
                    [sample for s, sample in zip(success, samples) if s == 1]
                )
                if sum(success) < N_samples:
                    logger.error(
                        f"The rountine found only {sum(success)} feasible state, but required are {N_samples}"
                    )
                logger.info(
                    f"Found {N_samples} feasible samples, tried {len(success)} samples"
                )
                return samples

    def solve_parametric_ocp_batch(self, x_batch, u_batch):
        """
        x_batch: np.array with shape (N_batch, nx)
        u_batch: np.array with shape (N_batch, nu)

        return Q_val_batch, grad_u_batch (N_batch, nu)
        """
        N_batch, _ = x_batch.shape
        Q_val_batch = np.empty((N_batch, 1))
        grad_x_batch = np.empty((N_batch, self.model.ns))
        grad_u_batch = np.empty((N_batch, self.model.na))

        if not self.parametric_solver_created:
            self._build_parametric_ocp()

        for batch_idx, (x0, u0) in enumerate(zip(x_batch, u_batch)):
            # set initial state
            # initialize solver? # TODO
            self.parametric_ocp_solver.load_iterate(
                "acados_def_iter/param_ocp_def_iter.json"
            )
            self.parametric_ocp_solver.set(0, "x", x0)
            self.parametric_ocp_solver.set(0, "u", u0)

            for i in range(self.N + 1):
                self.parametric_ocp_solver.set(i, "x", (1 - i / self.N + 1) * x0)
            for i in range(self.N):
                self.parametric_ocp_solver.set(
                    i,
                    "pi",
                    np.ones(
                        self.model.ns,
                    ),
                )

            self.parametric_ocp_solver.set(
                0, "lbx", x0
            )  # TODO wrong way to do it! I have to pass x0, u0, but I dont have to move the desired lb and ub
            self.parametric_ocp_solver.set(0, "ubx", x0)
            self.parametric_ocp_solver.set(0, "lbu", u0)
            self.parametric_ocp_solver.set(0, "ubu", u0)
            # solve
            solver_stats = {}
            solver_stats["return_status"] = self.parametric_ocp_solver.solve()
            value_cost_fn = self.parametric_ocp_solver.get_cost()

            # logger.info(f"Time tot [sec]: {self.parametric_ocp_solver.get_stats('time_tot')}, sqp_iter: {self.parametric_ocp_solver.get_stats('sqp_iter')}")
            if solver_stats["return_status"] != 0:
                # self.parametric_ocp_solver.print_statistics()
                logger.warning(
                    f"acados solver returned status = {solver_stats['return_status']}"
                )
                solver_stats["success"] = 0
                Q_val_batch[batch_idx, :] = np.nan
                grad_u_batch[batch_idx, :] = 0 * np.ones(self.model.na)
                grad_x_batch[batch_idx, :] = None
            else:
                solver_stats["success"] = 1
                # order: [ lbu lbx lg  ubu ubx ug  lsg usg]
                ny = self.model.na + self.model.ns
                lambda0 = self.parametric_ocp_solver.get(0, "lam")[
                    : self.model.na
                ]  # lbu
                lambda0 += -self.parametric_ocp_solver.get(0, "lam")[
                    2 * ny : 2 * ny + self.model.na
                ]  # ubu

                Q_val_batch[batch_idx, :] = value_cost_fn
                grad_u_batch[batch_idx, :] = lambda0
                grad_x_batch[batch_idx, :] = None

        return Q_val_batch, grad_u_batch, grad_x_batch

    def _build_parametric_quadratic_ocp(self) -> tuple:
        """parameters: x0, u0 -> solve for u1, lam_u0"""
        N = self.N
        # Formulate the ocp-nlp
        quad_parametric_ocp = AcadosOcp()
        tmp_dir = tempfile.mkdtemp(prefix="acados_quad_param_ocp_", dir=self.c_code_dir)
        quad_parametric_ocp.code_export_directory = tmp_dir
        quad_parametric_ocp.model.x = self.model.s
        quad_parametric_ocp.model.u = self.model.a
        quad_parametric_ocp.model.f_expl_expr = self.model.s_dot
        quad_parametric_ocp.model.name = (
            self.model.name
        )  # + "_quad_parametric" + "_" + str(time.time_ns())

        nx = self.model.ns
        nu = self.model.na
        ny = nx + nu
        quad_parametric_ocp.dims.N = N
        # cost
        quad_parametric_ocp.cost.cost_type = "NONLINEAR_LS"
        quad_parametric_ocp.cost.cost_type_e = "NONLINEAR_LS"
        quad_parametric_ocp.model.cost_y_expr = ca.vertcat(self.model.s, self.model.a)
        quad_parametric_ocp.model.cost_y_expr_e = self.model.s
        quad_parametric_ocp.cost.yref = np.zeros(ny)
        quad_parametric_ocp.cost.yref_e = np.zeros(nx)
        quad_parametric_ocp.cost.W = LA.block_diag(self.model.Q, self.model.R) / N
        quad_parametric_ocp.cost.W_e = self.model.lin_model["P_N"] / N
        # setting solver options
        quad_parametric_ocp.solver_options.tf = self.model.dt * N
        quad_parametric_ocp.solver_options.nlp_solver_type = self.qp_solver_options[
            "nlp_solver_type"
        ]
        quad_parametric_ocp.solver_options.qp_solver = self.qp_solver_options[
            "qp_solver"
        ]
        quad_parametric_ocp.solver_options.qp_solver_iter_max = self.qp_solver_options[
            "qp_solver_iter_max"
        ]
        # quad_parametric_ocp.solver_options.levenberg_marquardt = self.qp_solver_options["levenberg_marquardt"]
        quad_parametric_ocp.solver_options.hessian_approx = self.qp_solver_options[
            "hessian_approx"
        ]
        quad_parametric_ocp.solver_options.integrator_type = self.qp_solver_options[
            "integrator_type"
        ]
        quad_parametric_ocp.solver_options.nlp_solver_type = self.qp_solver_options[
            "nlp_solver_type"
        ]
        quad_parametric_ocp.solver_options.nlp_solver_max_iter = self.qp_solver_options[
            "nlp_solver_max_iter"
        ]
        quad_parametric_ocp.solver_options.tol = self.qp_solver_options["tol"]
        quad_parametric_ocp.solver_options.nlp_solver_tol_stat = (
            self.qp_solver_options["tol"] * 1e1
        )
        quad_parametric_ocp.solver_options.qp_solver_tol_eq = (
            self.qp_solver_options["tol"] * 1e-2
        )
        quad_parametric_ocp.solver_options.qp_solver_tol_ineq = (
            self.qp_solver_options["tol"] * 1e-1
        )
        quad_parametric_ocp.solver_options.qp_solver_tol_stat = (
            self.qp_solver_options["tol"] * 1e-1
        )
        quad_parametric_ocp.solver_options.qp_solver_tol_comp = (
            self.qp_solver_options["tol"] * 1e-1
        )
        # quad_parametric_ocp.solver_options.globalization = 'MERIT_BACKTRACKING'
        # quad_parametric_ocp.solver_options.globalization_use_SOC = 1
        # quad_parametric_ocp.solver_options.alpha_min = 0.2
        quad_parametric_ocp.solver_options.regularize_method = "PROJECT"

        # constraints
        quad_parametric_ocp.constraints.x0 = np.zeros(self.model.ns)
        quad_parametric_ocp.constraints.idxbu = np.arange(self.model.na)
        quad_parametric_ocp.constraints.lbu = -INF * np.ones([self.model.na])
        quad_parametric_ocp.constraints.ubu = INF * np.ones([self.model.na])

        # g constraint needed instead of bound on u because we need to slack u_0
        quad_parametric_ocp.constraints.C = np.vstack(
            [np.diag(np.ones(self.model.ns)), np.zeros((self.model.na, self.model.ns))]
        )
        quad_parametric_ocp.constraints.D = np.vstack(
            [np.zeros((self.model.ns, self.model.na)), np.diag(np.ones(self.model.na))]
        )
        quad_parametric_ocp.constraints.lg = np.concatenate(
            [self.model.s_min, self.model.a_min]
        )
        quad_parametric_ocp.constraints.ug = np.concatenate(
            [self.model.s_max, self.model.a_max]
        )
        quad_parametric_ocp.constraints.C_e = np.diag(np.ones(self.model.ns))
        quad_parametric_ocp.constraints.lg_e = self.model.s_min
        quad_parametric_ocp.constraints.ug_e = self.model.s_max
        # slacks
        quad_parametric_ocp.constraints.idxsg = np.arange(self.model.ns + self.model.na)
        quad_parametric_ocp.constraints.idxsg_e = np.arange(self.model.ns)
        quad_parametric_ocp.cost.Zl = (
            np.concatenate([self.weight_l2_x, self.weight_l2_u]) / N
        )
        quad_parametric_ocp.cost.zl = (
            np.concatenate([self.weight_l1_x, self.weight_l1_u]) / N
        )
        quad_parametric_ocp.cost.Zl_e = self.model.dt * self.weight_l2_x / N
        quad_parametric_ocp.cost.zl_e = self.model.dt * self.weight_l1_x / N
        quad_parametric_ocp.cost.Zu = (
            np.concatenate([self.weight_l2_x, self.weight_l2_u]) / N
        )
        quad_parametric_ocp.cost.zu = (
            np.concatenate([self.weight_l1_x, self.weight_l1_u]) / N
        )
        quad_parametric_ocp.cost.Zu_e = self.model.dt * self.weight_l2_x / N
        quad_parametric_ocp.cost.zu_e = self.model.dt * self.weight_l1_x / N

        if self.no_cython:
            self.quad_parametric_ocp_solver = AcadosOcpSolver(
                quad_parametric_ocp, json_file=tmp_dir + "/acados_ocp_nlp.json"
            )
        else:
            solver_json = tmp_dir + "/acados_ocp_nlp.json"
            AcadosOcpSolver.generate(quad_parametric_ocp, json_file=solver_json)
            AcadosOcpSolver.build(
                quad_parametric_ocp.code_export_directory, with_cython=True
            )
            self.quad_parametric_ocp_solver = AcadosOcpSolver.create_cython_solver(
                solver_json
            )
        # self.quad_parametric_ocp_solver.store_iterate(filename="acados_def_iter/quad_param_ocp_def_iter.json", overwrite=True)
        self.qp_solver_created = True

    def solve_parametric_quadratic_ocp(
        self, x0: np.ndarray, u0: np.ndarray, lin_vector: dict
    ) -> dict:
        if not self.qp_solver_created:
            self._build_parametric_quadratic_ocp()

        states = np.zeros((self.N + 1, self.model.ns))
        controls = np.zeros((self.N, self.model.na))

        lin_vector = copy.deepcopy(lin_vector)
        # slacks coming from standard ocp where only state is slacked (n_slack=nx), here n_slack = nx+nu but only 0,...,N-1
        for k in lin_vector["slacks"].keys():
            lin_vector["slacks"][k] = np.hstack(
                [lin_vector["slacks"][k], np.zeros((self.N + 1, self.model.na))]
            )

        # TODO: load default iterate?
        # set linearization vector
        for i in range(self.N):
            self.quad_parametric_ocp_solver.set(i, "u", lin_vector["controls"][i, :])
            self.quad_parametric_ocp_solver.set(i, "x", lin_vector["states"][i, :])
        self.quad_parametric_ocp_solver.set(
            self.N, "x", lin_vector["states"][self.N, :]
        )
        for i in range(1, self.N):
            self.quad_parametric_ocp_solver.set(
                i, "sl", lin_vector["slacks"]["lower"][i, :]
            )  # TODO: slack linearization!
            self.quad_parametric_ocp_solver.set(
                i, "su", lin_vector["slacks"]["upper"][i, :]
            )
        self.quad_parametric_ocp_solver.set(
            self.N, "sl", lin_vector["slacks"]["lower"][self.N, : self.model.ns]
        )  # TODO: slack linearization!
        self.quad_parametric_ocp_solver.set(
            self.N, "su", lin_vector["slacks"]["upper"][self.N, : self.model.ns]
        )

        # set (x0, u0) to evaluate
        self.quad_parametric_ocp_solver.set(0, "lbx", x0)
        self.quad_parametric_ocp_solver.set(0, "ubx", x0)
        self.quad_parametric_ocp_solver.set(0, "lbu", u0)
        self.quad_parametric_ocp_solver.set(0, "ubu", u0)

        # solve
        solver_stats = {}
        solver_stats["return_status"] = self.quad_parametric_ocp_solver.solve()
        value_cost_fn = self.quad_parametric_ocp_solver.get_cost()

        logger.info(
            f"Time tot [sec]: {self.quad_parametric_ocp_solver.get_stats('time_tot')}, sqp_iter: {self.quad_parametric_ocp_solver.get_stats('sqp_iter')}"
        )
        if solver_stats["return_status"] != 0:
            # self.parametric_ocp_solver.print_statistics()
            logger.warning(
                f"acados solver returned status = {solver_stats['return_status']}"
            )
            solver_stats["success"] = 0
        else:
            solver_stats["success"] = 1

        for i in range(self.N + 1):
            states[i, :] = self.quad_parametric_ocp_solver.get(i, "x")
        for i in range(self.N):
            controls[i, :] = self.quad_parametric_ocp_solver.get(i, "u")

        # order: [ lbu lbx lg  ubu ubx ug  lsg usg]
        ny = self.model.na + self.model.ns
        lambda0 = self.quad_parametric_ocp_solver.get(0, "lam")[: self.model.na]  # lbu
        lambda0 += -self.quad_parametric_ocp_solver.get(0, "lam")[
            2 * ny : 2 * ny + self.model.na
        ]  # ubu
        # logger.info(f"{lambda0=},")

        return {
            "value_cost_fn": value_cost_fn,
            "states": states,
            "controls": controls,
            "solver_stats": solver_stats,
            "lambda0": lambda0,
        }

    def solve_parametric_quadratic_ocp_batch(self, x_batch, u_batch, lin_list: list):
        """
        x_batch: np.array with shape (N_batch, nx)
        u_batch: np.array with shape (N_batch, nu)
        lin_list: list of list with linearization vector with order [states, controls, slacks_lower, slacks_upper]

        return Q_val_batch, grad_u_batch (N_batch, nu)
        """
        N_batch, _ = x_batch.shape
        Q_val_batch = np.empty((N_batch, 1))
        grad_x_batch = np.empty((N_batch, self.model.ns))
        grad_u_batch = np.empty((N_batch, self.model.na))
        lin_list = copy.deepcopy(lin_list)
        if not self.qp_solver_created:
            self._build_parametric_quadratic_ocp()

        self.quad_parametric_ocp_solver.load_iterate(
            "acados_def_iter/quad_param_ocp_def_iter.json"
        )
        for batch_idx, (x0, u0) in enumerate(zip(x_batch, u_batch)):
            # TODO: order the batch such that if there are same x0 they become consecutive and we avoid to set again the initialization!
            # slacks coming from standard ocp where only state is slacked (n_slack=nx), here n_slack = nx+nu but only 0,...,N-1
            lin_vec_slacks_l = np.hstack(
                [lin_list[2][batch_idx, :, :], np.zeros((self.N + 1, self.model.na))]
            )
            lin_vec_slacks_u = np.hstack(
                [lin_list[3][batch_idx, :, :], np.zeros((self.N + 1, self.model.na))]
            )
            lin_vec_states = np.array(lin_list[0][batch_idx, :, :])
            lin_vec_controls = np.array(lin_list[1][batch_idx, :, :])

            # set linearization vector
            # TODO: load default iterate?
            for i in range(self.N):
                self.quad_parametric_ocp_solver.set(i, "u", lin_vec_controls[i, :])
                self.quad_parametric_ocp_solver.set(i, "x", lin_vec_states[i, :])
            self.quad_parametric_ocp_solver.set(self.N, "x", lin_vec_states[self.N, :])
            for i in range(1, self.N):
                self.quad_parametric_ocp_solver.set(
                    i, "sl", lin_vec_slacks_l[i, :]
                )  # TODO: slack linearization!
                self.quad_parametric_ocp_solver.set(i, "su", lin_vec_slacks_u[i, :])
            self.quad_parametric_ocp_solver.set(
                self.N, "sl", lin_vec_slacks_l[self.N, : self.model.ns]
            )  # TODO: slack linearization!
            self.quad_parametric_ocp_solver.set(
                self.N, "su", lin_vec_slacks_u[self.N, : self.model.ns]
            )

            # set (x0, u0) to evaluate
            self.quad_parametric_ocp_solver.set(0, "lbx", x0)
            self.quad_parametric_ocp_solver.set(0, "ubx", x0)
            self.quad_parametric_ocp_solver.set(0, "lbu", u0)
            self.quad_parametric_ocp_solver.set(0, "ubu", u0)

            # solve
            solver_stats = {}
            solver_stats["return_status"] = self.quad_parametric_ocp_solver.solve()
            value_cost_fn = self.quad_parametric_ocp_solver.get_cost()

            # logger.info(f"Time tot [sec]: {self.quad_parametric_ocp_solver.get_stats('time_tot')}, sqp_iter: {self.quad_parametric_ocp_solver.get_stats('sqp_iter')}")
            if solver_stats["return_status"] != 0:
                # self.quad_parametric_ocp_solver.print_statistics()
                logger.warning(
                    f"acados solver returned status = {solver_stats['return_status']}"
                )
                solver_stats["success"] = 0
                Q_val_batch[batch_idx, :] = np.nan
                grad_u_batch[batch_idx, :] = 0 * np.ones(self.model.na)
                grad_x_batch[batch_idx, :] = None
            else:
                solver_stats["success"] = 1
                # order: [ lbu lbx lg  ubu ubx ug  lsg usg]
                ny = self.model.na + self.model.ns
                lambda0 = self.quad_parametric_ocp_solver.get(0, "lam")[
                    : self.model.na
                ]  # lbu
                lambda0 += -self.quad_parametric_ocp_solver.get(0, "lam")[
                    2 * ny : 2 * ny + self.model.na
                ]  # ubu

                Q_val_batch[batch_idx, :] = value_cost_fn
                grad_u_batch[batch_idx, :] = lambda0
                grad_x_batch[batch_idx, :] = None

        return Q_val_batch, grad_u_batch, grad_x_batch
