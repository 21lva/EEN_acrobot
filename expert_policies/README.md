# Expert Policies

Train & test your expert policies (TRPO), and save trajectories made by the trained policies for testing EEN or world model. 

## To train a policy with TRPO:
By going to the `/your-path-to-LG-dir/LG_Lamination/expert_policies` directory and executing the command below, the policy will be saved (overwritten) to the `LG_Lamination/expert_policies/saved_policies/<environment_id>` directory for every 100 episodes during the training.

Also, if you are using a multi-gpu system, you can specify the gpu that you want to use. If you have 4 gpus, than the gpu ids will be 0,1,2,3.

Currently, policy network with 2 hidden layers with 64 neurons for each layer are used.
```bash
python trpo_train --env=<environment_id> --gpu=<gpu_id> --render=<bool>
```

## To evaluate the saved policy and generate trajectories from it.
By going to the `/your-path-to-LG-dir/LG_Lamination/expert_policies` directory and entering the command below loads the policy saved in `LG_Lamination/expert_policies/saved_policies/<environment_id>` and prints out the average reward from trials. The number of trials can be set by writing numbers to the `--n_trials` option.
If you want to save your trajectories as npz file, then write `True` for the `--save_traj` option in the command written below. Same goes for the `--render` which enables/disables rendering.

```bash
python trpo_test --env=<environment_id> --gpu=<gpu_id> --render=<bool> --save_traj=<bool> --n_trials=<integer> 

```

## Saved trajectories

The npz file saved in `LG_Lamination/expert_policies/saved_trajectories` contain npz files

#### Data format
The npz files contains 1-dim numpy array which dtype is "object" since its each element contains the information of an episode (trajectory) with **different** number of transitions.

Therefore,
```bash
>>> import numpy as np
>>> trajs=np.load("./saved_trajectories/trpo_Acrobot-v1/Acrobot-v1_1000_trials.npz")
>>> trajs["obs"].shape
(1000,)
>>> trajs["act"].shape
(1000,)
>>> trajs["rew"].shape
(1000,)

```

The above result comes out since I have saved 1000 trajectories.

For the shape of each element of the numpy array `trajs`,
```bash
>>> traj["obs"][0].shape
(88, 6)
>>> traj["act"][0].shape
(88, 1)
>>> traj["rew"][0].shape
(88, 1)
>>> traj["obs"][1].shape
(85, 6)
>>> traj["act"][1].shape
(85, 1)
>>> traj["rew"][1].shape
(85, 1)

``` 

The above result is due to the 1st/2nd episode saved in the `trajs` having 88/85 transitons, 6-dim state observation, and 1-dim action space.  

## Hyperparameters for TRPO

Difference between the settings that I used in Acrobot-v1 and Reacher-v2 is the `max_kl`.

### TRPO - Acrobot-v1
```bash
network=mlp(num_hidden=64, num_layers=2),  
total_timesteps=int(1e7),
timesteps_per_batch=1024,
max_kl=0.001, 
cg_iters=10,
gamma=0.99,
lam=0.98,
seed=None,
ent_coef=0.0,
cg_damping=0.1,
vf_stepsize=1e-3,
vf_iters=5,
callback=None,
load_path=None,
save_interval = 100,
test_run=False,
normalize_observations = True

```


### TRPO - Reacher-v2
```bash
network=mlp(num_hidden=64, num_layers=2),  
total_timesteps=int(1e7),
timesteps_per_batch=1024,
max_kl=0.01, 
cg_iters=10,
gamma=0.99,
lam=0.98,
seed=None,
ent_coef=0.0,
cg_damping=0.1,
vf_stepsize=1e-3,
vf_iters=5,
callback=None,
load_path=None,
save_interval = 100,
test_run=False,
normalize_observations = True
```


## Performance of the saved policies
For this evaluation, the rewards are summed across transitions for each episodes (return without discount factor across time) and averaged over 1000 episodes.
### TRPO-Acrobot-v1
Average rewards across 1000 episodes: -80.692
### TRPO-Reacher-v2
Average rewards across 1000 episodes: -5.586
