from itertools import chain, product

from tqdm.contrib.concurrent import process_map

from imitate.main_training import create_argument_parser, main_train


def main_sweep():
    net_width = [64, 128, 256]
    net_depth = [1, 2, 3]
    seeds = [0, 1, 2]
    learning_rates = [1e-2, 1e-3]
    # batch_size = [16, 32, 64]
    num_training_steps = [2000]

    l2_iter = product(net_width, net_depth, seeds, learning_rates, num_training_steps, [False], ["l2"])
    vfn_iter = product(net_width, net_depth, seeds, learning_rates, num_training_steps, [False, True], ["vfn"])
    comb_iter = chain(l2_iter, vfn_iter)

    def args_iter():
        for width, depth, seed, lr, steps, qa, loss in comb_iter:
            args = create_argument_parser()
            args.qp_approx = qa
            args.loss = loss
            args.nn_width = width
            args.nn_depth = depth
            args.seed = seed
            args.learning_rate = lr
            args.num_training_steps = steps
            args.save_net = True
            args.exp_dir = f"experiments/cartpole/hp_sweep/loss_{loss}_width_{width}_depth_{depth}_lr_{lr}_seed_{seed}_qpapprox_{qa}"
            yield args

    # Use tqdm to show progress and multiprocessing to speed up
    process_map(main_train, args_iter(), max_workers=8)


if __name__ == "__main__":
    main_sweep()
