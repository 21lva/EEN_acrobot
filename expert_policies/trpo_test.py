import os
import gym
import baselines.common.tf_util as tf_util
import baselines.common.tests.util as test_util
import tensorflow as tf, numpy as np
import baselines.trpo_mpi.trpo_mpi as trpo
import utils.tools as tools
from baselines.common.models import mlp #, cnn_small
import numpy as np


def load_trajectories(file_name):
    data = np.load(file_name)
    return data

def gen_trajectories(env, env_id, policy, n_trials, do_render, save_traj):

    observations, actions, rewards = test_util.rollout(env=env, model=policy, n_trials=n_trials, render_flag = do_render)

    if save_traj is True:
        ep_obs_list = []
        ep_act_list = []
        ep_rew_list = []

        assert (n_trials == len(observations)), "length of observations list should be equal to the number of trials"

        for i in range(n_trials):
            ep_obs_list.append(np.vstack(observations[i]))
            ep_act_list.append(np.vstack(actions[i]))
            ep_rew_list.append(np.vstack(rewards[i]))

        # ep_obj_np      shape: (N_TRIALS,) / dtype: object.
        # ep_obs_list[0] shape: [# of transitions in the 1st ep]x[# of observation values] / dtype: float64
        # every ep_obs_list[i] (varying "i") has different # of transitions.
        ep_obs_np = np.asarray(ep_obs_list)
        ep_act_np = np.asarray(ep_act_list)
        ep_rew_np = np.asarray(ep_rew_list)

        save_traj_path = './saved_trajectories/trpo_' + env_id
        tools.make_dir_if_none(save_traj_path)
        np.savez_compressed(os.path.join(save_traj_path, env_id+"_"+str(n_trials)+"_trials.npz"), obs=ep_obs_np, act=ep_act_np,
                            rew=ep_rew_np)
    return rewards

def get_avg_rewards(rewards):
    rew_sum = [sum(r) for r in rewards]
    rew_avg = sum(rew_sum) / len(rewards)
    print("#####################################################################")
    print("Average reward in {} episodes is {}".format(len(rewards), rew_avg))
    print("#####################################################################")



if __name__ == '__main__':

    args = tools.parse_input_str()
    env_id = args.env_id
    do_render = args.do_render
    gpu_id = args.gpu_id
    save_traj = args.save_traj
    n_trials = args.n_trials

    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    print("Using GPU : " + gpu_id)


    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True

    with tf.Session(config=sess_config):
        env = gym.make(env_id)
        env.seed(0)

        # make tensorflow graph before loading the policy
        policy = trpo.learn(network=mlp(num_hidden=64, num_layers=2),
                   env=env,
                   total_timesteps=int(1e7),
                   max_kl=0.001,
                   cg_iters=10,
                   gamma=0.99,
                   lam=0.99,
                   ent_coef=0.0,
                   pi_save_path = None,
                   test_run=True)

        # load policy
        policy_load_path = './saved_policies/trpo_' + env_id + '/policy'
        tf_util.load_state(policy_load_path, sess=None)
        print("Policy will be loaded from: [" + './saved_policies/trpo_' + env_id + "]")

        rewards = gen_trajectories(env=env, env_id=env_id, policy=policy, n_trials=n_trials, do_render = do_render, save_traj=save_traj)
        get_avg_rewards(rewards)



