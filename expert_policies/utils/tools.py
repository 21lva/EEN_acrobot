import os
import argparse


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_input_str():
    parser = argparse.ArgumentParser(description='Environment ID, GPU ID')
    parser.add_argument('--env', dest='env_id', action='store', default='Acrobot-v1',
                        help="Write OpenAI gym environment ID. ex) 'Acrobot-v1'")
    parser.add_argument('--gpu', dest='gpu_id', action='store', default='0',
                        help="Write GPU ID. ex) '0' ")
    parser.add_argument('--n_trials', dest='n_trials', action='store', type = int, default=100,
                        help="Number of test trials for saving or evaluating the saved policy")
    parser.add_argument("--render", dest='do_render', action='store', type=str2bool,
                        nargs='?', const=True, default=False,
                        help="Do rendering? [True/False].")
    parser.add_argument("--save_traj", dest='save_traj', action='store', type=str2bool,
                        nargs='?', const=True, default=False,
                        help="save_trajectories? [True/False].")
    args = parser.parse_args()
    return args


def make_dir_if_none(dir_path):
    if os.path.exists(dir_path):
        print("The directory ["+dir_path+"] exists")
    else:
        os.makedirs(dir_path)


