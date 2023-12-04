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
