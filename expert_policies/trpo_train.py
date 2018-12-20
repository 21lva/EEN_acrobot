#!/usr/bin/env python

"""
Train TRPO on a Acrobot-v1 to get expert policy & trajectories
"""


import tensorflow as tf
import gym
import baselines.trpo_mpi.trpo_mpi as trpo
import utils.tools as tools
from baselines.common.models import mlp #, cnn_small

############################################
# Select which GPU to use
import os

############################################

def main():

    args = tools.parse_input_str()
    # env_id: 'Acrobot-v1', 'Reacher-v2'
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    policy_save_path = './saved_policies/trpo_'+args.env_id+'/policy'
    tools.make_dir_if_none('./saved_policies/trpo_'+args.env_id)

    print("Using GPU               : "+args.gpu_id)
    print("Training on environment : "+args.env_id)
    print("Policy will be saved in : ["+policy_save_path+"]")

    """Run TRPO"""
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True # pylint: disable=E1101
    with tf.Session(config=sess_config):

        # Some of the settings are from the baselines/baselines/trpo_mpi/defaults.py
        trpo.learn(network=mlp(num_hidden=64, num_layers=2), #'mlp',# baselines.common/policies.py/build_policy
                   env=gym.make(args.env_id),
                   total_timesteps=int(1e7),
                   timesteps_per_batch=1024,
                   max_kl=0.001, #0.01 for the Reacher 0.001 for the Acrobot
                   cg_iters=10,
                   gamma=0.99,
                   lam=0.98,
                   seed=None,
                   ent_coef=0.0,
                   cg_damping=0.1,
                   vf_stepsize=1e-3,
                   vf_iters=5,
                   # max_episodes=0,
                   # max_iters=0,  # time constraint
                   callback=None,
                   load_path=None,
                   pi_save_path=policy_save_path, #'./saved_policies/trpo_Acrobot-v1',
                   save_interval = 100,
                   test_run=False,
                   normalize_observations = True)  # baselines/baselines/common.policies.py _normalize_clip_observation()

if __name__ == '__main__':
    main()

