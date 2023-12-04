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

from itertools import chain, product

from tqdm.contrib.concurrent import process_map

from imitate.main_training import create_argument_parser, main_train


def main_sweep():
    net_width = [128]
    net_depth = [2]
    seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    num_training_steps = [5001]

    vfn_iter = product([False, True], ["vfn"], net_width, net_depth, seeds, num_training_steps)
    l2_iter = product([False], ["l2"], net_width, net_depth, seeds, num_training_steps)
    comb_iter = chain(vfn_iter, l2_iter)

    def args_iter():
        for qa, loss, width, depth, seed, steps in comb_iter:
            args = create_argument_parser()
            args.save_checkpoint = True
            args.qp_approx = qa
            args.loss = loss
            args.nn_width = width
            args.nn_depth = depth
            args.seed = seed
            args.num_training_steps = steps
            args.save_net = True
            args.exp_dir = f"experiments/cartpole/conv_sweep/seed_{seed}_loss_{loss}_qpapprox_{qa}"
            yield args

    # Use tqdm to show progress and multiprocessing to speed up
    process_map(main_train, args_iter(), max_workers=15)


if __name__ == "__main__":
    main_sweep()
