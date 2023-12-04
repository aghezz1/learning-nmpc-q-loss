import copy
import casadi as ca
import logging
import numpy as np
import scipy.linalg as LA
from abc import ABC, abstractmethod

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# TODO: weight for the cartpole, import them from/to mpc.py
weight_l1_x = np.array([10, 1, 10, 1])
weight_l2_x = 2 * np.array([1e3, 1e2, 1e3, 1e2])
weight_l1_u = np.array([1e5])
weight_l2_u = 2 * np.array([1e4])


class MpcModelBase(ABC):
    # expose state, control, dynamics (s_dot)
    # provide integrators
    @property
    def scaling_factor_action(self):
        try:
            return self._scaling_factor_action
        except:
            logger.warning(
                "scaling_factor_action not defined yet, returning 1 instead!"
            )
            return 1.0

    def _integrate_rk4(self, m_steps: int = None):
        """
        m_steps: number of integration steps per interval
        """
        if m_steps is None:
            m_steps = copy.deepcopy(self.m_steps)
        dt = self.dt / m_steps
        if self.tracking_cost_flag:
            f = ca.Function("f", [self.s, self.a], [self.s_dot, self.tracking_cost])
            X0 = ca.SX.sym("X0", self.ns)
            U = ca.SX.sym("U", self.na)
            X = X0
            Q = 0
            for j in range(m_steps):
                k1, k1_q = f(X, U)
                k2, k2_q = f(X + dt / 2 * k1, U)
                k3, k3_q = f(X + dt / 2 * k2, U)
                k4, k4_q = f(X + dt * k3, U)
                X = X + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
                Q = Q + dt / 6 * (k1_q + 2 * k2_q + 2 * k3_q + k4_q)
            return ca.Function("I_rk4", [X0, U], [X, Q], ["x0", "u"], ["xf", "qf"])
        else:
            f = ca.Function("f", [self.s, self.a], [self.s_dot])
            X0 = ca.SX.sym("X0", self.ns)
            U = ca.SX.sym("U", self.na)
            X = X0
            for j in range(m_steps):
                k1 = f(X, U)
                k2 = f(X + dt / 2 * k1, U)
                k3 = f(X + dt / 2 * k2, U)
                k4 = f(X + dt * k3, U)
                X = X + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            return ca.Function("I_rk4", [X0, U], [X], ["x0", "u"], ["xf"])

    def _integrate_explicit_euler(self):
        if self.tracking_cost_flag:
            f = ca.Function("f", [self.s, self.a], [self.s_dot, self.tracking_cost])
            X0 = ca.SX.sym("X0", self.ns)
            U = ca.SX.sym("U", self.na)
            X = X0
            Q = 0
            k1, q1 = f(X, U)
            X = X + self.dt * k1
            Q = Q + self.dt * q1
            return ca.Function("I_ee", [X0, U], [X, Q], ["x0", "u"], ["xf", "qf"])
        else:
            f = ca.Function("f", [self.s, self.a], [self.s_dot])
            X0 = ca.SX.sym("X0", self.ns)
            U = ca.SX.sym("U", self.na)
            X = X0
            k1 = f(X, U)
            X = X + self.dt * k1
            return ca.Function("I_ee", [X0, U], [X], ["x0", "u"], ["xf"])

    def _set_integrator(self, integrator, dt, m_steps):
        self.integrator_type = integrator
        self.dt = dt
        self.m_steps = m_steps
        if integrator == "rk4":
            self.integrator = self._integrate_rk4()
        elif integrator == "ee":
            self.integrator = self._integrate_explicit_euler()
        else:
            raise AttributeError(
                "The integrator can be rk4 (runge-kutta 4) or ee (explicit euler)!"
            )

    def create_auxiliary_functions(self):
        integrator_sx_eval = self.integrator(x0=self.s, u=self.a)

        self.f_dyn = integrator_sx_eval["xf"]
        self.f_dyn_fn = ca.Function("f_dyn", [self.s, self.a], [self.f_dyn])
        f_dyn_dot = ca.jacobian(self.f_dyn, ca.vertcat(self.a, self.s))
        self.f_dyn_dot_fn = ca.Function("f_dyn_dot", [self.s, self.a], [f_dyn_dot])

        self.stage_cost = integrator_sx_eval["qf"]
        self.stage_cost_dot = ca.jacobian(self.stage_cost, ca.vertcat(self.s, self.a))
        self.stage_cost_fn = ca.Function(
            "stage_cost", [self.s, self.a], [self.stage_cost]
        )
        self.stage_cost_dot_fn = ca.Function(
            "stage_cost_dot", [self.s, self.a], [self.stage_cost_dot]
        )

        def _constr_cost_l1(var, var_min, var_max, weight):
            return self.dt * ca.sum1(weight * (ca.fmax(var - var_max, 0) + ca.fmax(var_min - var, 0)))
        def _constr_cost_l2(var, var_min, var_max, weight):
            return self.dt * 0.5 * ca.sum1(weight * (ca.fmax(var - var_max, 0)**2 + ca.fmax(var_min - var, 0)**2))

        self.constr_cost = _constr_cost_l1(self.s, self.s_min, self.s_max, weight_l1_x)
        self.constr_cost += _constr_cost_l1(self.a, self.a_min, self.a_max, weight_l1_u)
        self.constr_cost += _constr_cost_l2(self.s, self.s_min, self.s_max, weight_l2_x)
        self.constr_cost += _constr_cost_l2(self.a, self.a_min, self.a_max, weight_l2_u)
        self.constr_cost_fn = ca.Function(
            "constr_cost", [self.s, self.a], [self.constr_cost]
        )
        constr_cost_dot = ca.jacobian(self.constr_cost, ca.vertcat(self.s, self.a))
        self.constr_cost_dot_fn = ca.Function(
            "constr_cost_dot", [self.s, self.a], [constr_cost_dot]
        )

    @abstractmethod
    def set_constraints_ocp(
        self,
        s_min: np.ndarray = None,
        s_max: np.ndarray = None,
        a_min: np.ndarray = None,
        a_max: np.ndarray = None,
    ):
        pass

    @abstractmethod
    def set_objective_ocp(self, Q: np.ndarray = None, R: np.ndarray = None):
        pass


class CartpoleModel(MpcModelBase):
    @property
    def tracking_cost_flag(self):
        try:
            return self._tracking_cost_flag
        except:
            logger.warning("tracking_cost not defined yet, returning None instead!")
            return False

    def _set_dynamics(self):
        l, m, M = 0.8, 0.1, 1.0
        g = 9.81
        self._scaling_factor_action = 25.0
        # Input variables.
        x = ca.SX.sym("x")
        x_dot = ca.SX.sym("x_dot")
        theta = ca.SX.sym("theta")
        theta_dot = ca.SX.sym("theta_dot")
        self.s = ca.vertcat(x, x_dot, theta, theta_dot)
        self.a = ca.SX.sym("U")
        self.ns = self.s.shape[0]
        self.na = self.a.shape[0]
        # Dynamics.
        cos_theta = ca.cos(theta)
        sin_theta = ca.sin(theta)
        denominator = M + m - m * cos_theta * cos_theta
        self.s_dot = ca.vertcat(
            x_dot,
            (
                -m * l * sin_theta * theta_dot * theta_dot
                + m * g * cos_theta * sin_theta
                + self.a * self.scaling_factor_action
            )
            / denominator,
            theta_dot,
            (
                -m * l * cos_theta * sin_theta * theta_dot * theta_dot
                + self.a * cos_theta * self.scaling_factor_action
                + (M + m) * g * sin_theta
            )
            / (l * denominator),
        )
        self.lin_model = {}
        self.lin_model["A"] = ca.Function("A_lin", [self.s, self.a], [ca.jacobian(self.s_dot, self.s)])
        self.lin_model["B"] = ca.Function("B_lin", [self.s, self.a], [ca.jacobian(self.s_dot, self.a)])

    def __init__(
        self, integrator: str = "rk4", dt: float = 0.05, m_steps: int = 1
    ) -> None:
        super().__init__()
        self.name = "cartpole"
        self._set_dynamics()
        self._set_integrator(integrator=integrator, dt=dt, m_steps=m_steps)

    def set_constraints_ocp(
        self,
        s_min: np.ndarray = None,
        s_max: np.ndarray = None,
        a_min: np.ndarray = None,
        a_max: np.ndarray = None,
    ):
        # Default values
        self.s_max = np.array([2, 4, np.pi / 3, 2])
        self.s_min = -self.s_max
        self.a_max = np.array([1])
        self.a_min = -self.a_max
        # User defined values
        if s_min is not None:
            self.s_min = s_min
        if s_max is not None:
            self.s_max = s_max
        if a_min is not None:
            self.a_min = a_min
        if a_max is not None:
            self.a_max = a_max

    def set_objective_ocp(self, Q: np.ndarray = None, R: np.ndarray = None):
        self.Q = np.diag([1e1, 1e0, 1e1, 1e0])
        self.R = np.diag([1e-1])
        if Q is not None:
            self.Q = Q
        if R is not None:
            self.R = R
        x_lin = np.zeros(self.ns)
        u_lin = np.zeros(self.na)
        self.lin_model["P_N"] = LA.solve_continuous_are(
            self.lin_model["A"](x_lin, u_lin),
            self.lin_model["B"](x_lin, u_lin),
            self.Q,
            self.R,
        )
        self.tracking_cost = 0.5 * (
            self.s.T @ self.Q @ self.s + self.a.T @ self.R @ self.a
        )
        self._tracking_cost_flag = True

        # update integrator with tracking cost
        self._set_integrator(
            integrator=self.integrator_type, dt=self.dt, m_steps=self.m_steps
        )
        logger.warning("tracking cost is now defined by the MPCBaseClass")


class InvertedPendulumModel(MpcModelBase):
    @property
    def tracking_cost_flag(self):
        try:
            return self._tracking_cost_flag
        except:
            logger.warning("tracking_cost not defined yet, returning None instead!")
            return False

    def _set_dynamics(self):
        theta = ca.SX.sym("theta")
        omega = ca.SX.sym("omega")

        self.s = ca.vertcat(theta, omega)  # state
        self.a = ca.SX.sym("a")  # control

        self.ns = self.s.shape[0]
        self.na = self.a.shape[0]

        # Setting dynamics
        theta_dot = omega
        omega_dot = (
            ca.sin(theta) + self._scaling_factor_action * self.a
        )  # - 0.2*omega #add damping
        self.s_dot = ca.vertcat(theta_dot, omega_dot)

    def __init__(self, integrator: str = "rk4", dt: float = 0.05, m_steps: int = 1):
        self.name = "inverted_pendulum"
        self._scaling_factor_action = 10.0
        self._set_dynamics()
        self._set_integrator(integrator=integrator, dt=dt, m_steps=m_steps)

    def set_constraints_ocp(
        self,
        s_min: np.ndarray = None,
        s_max: np.ndarray = None,
        a_min: np.ndarray = None,
        a_max: np.ndarray = None,
    ):
        # Default values
        self.s_min = np.array([-np.pi, -1.0])
        self.s_max = -self.s_min
        self.a_max = np.array([1])
        self.a_min = -self.a_max
        # User defined values
        if s_min is not None:
            self.s_min = s_min
        if s_max is not None:
            self.s_max = s_max
        if a_min is not None:
            self.a_min = a_min
        if a_max is not None:
            self.a_max = a_max

    def set_objective_ocp(self, Q: np.ndarray = None, R: np.ndarray = None):
        self.Q = np.diag([100, 0.01])
        self.R = np.diag([0.05 * self.scaling_factor_action**2])
        if Q is not None:
            self.Q = Q
        if R is not None:
            self.R = R

        self.tracking_cost = 0.5 * (
            self.s.T @ self.Q @ self.s + self.a.T @ self.R @ self.a
        )
        self._tracking_cost_flag = True

        # update integrator with tracking cost
        self._set_integrator(
            integrator=self.integrator_type, dt=self.dt, m_steps=self.m_steps
        )
        logger.warning("tracking cost is now defined by the MPCBaseClass")
