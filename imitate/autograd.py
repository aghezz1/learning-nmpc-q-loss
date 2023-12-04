import logging

import numpy as np
import torch
from torch import autograd

from imitate.mpc_acados import MPCBaseClass
from imitate.models import InvertedPendulumModel

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class QMpc(autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        u: torch.Tensor,
        ocp_solver: MPCBaseClass,
        qp_approx: bool = False,
        lin_vec: torch.Tensor = None,
    ):
        device, dtype = x.device, x.dtype

        x_cpu = x.cpu()
        u_cpu = u.cpu()
        ctx.save_for_backward(x_cpu, u_cpu)

        x_np = x_cpu.detach().numpy().astype("float64")
        u_np = u_cpu.detach().numpy().astype("float64")
        if qp_approx:
            (
                Q_val_batch,
                grad_u_batch,
                grad_x_batch,
            ) = ocp_solver.solve_parametric_quadratic_ocp_batch(x_np, u_np, lin_vec)
        else:
            (
                Q_val_batch,
                grad_u_batch,
                grad_x_batch,
            ) = ocp_solver.solve_parametric_ocp_batch(x_np, u_np)
        ctx.grad_u_batch = grad_u_batch
        ctx.grad_x_batch = grad_x_batch

        return torch.tensor(np.nanmean(Q_val_batch), device=device, dtype=dtype)

    @staticmethod
    def backward(ctx, grad_output):
        device, dtype = grad_output.device, grad_output.dtype

        grad_u_batch = torch.tensor(ctx.grad_u_batch, device=device, dtype=dtype)
        grad_x_batch = torch.tensor(ctx.grad_x_batch, device=device, dtype=dtype)

        return grad_x_batch, grad_u_batch, None, None, None


if __name__ == "__main__":

    u = torch.zeros((32, 1), requires_grad=True)
    x = torch.zeros((32, 2), requires_grad=False)

    model = InvertedPendulumModel()

    mpc = MPCBaseClass(model, 20)

    QMpc.apply(x, u, mpc)
