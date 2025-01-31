# SGPbrl

## Subgoal-Guided Reward Shaping: Improving Preference-Based Offline Reinforcement Learning via Conditional VAEs

## NOTICE
We implement our code based on [Preference Transformer](https://github.com/csmile-1006/PreferenceTransformer). 
Our model is built upon the Preference Transformer framework. Therefore, by running the Preference Transformer, you can obtain our results.

## Table of Contents
- [Installation](#installation)
- [How to Run the Code](#how-to-run-the-code)
  - [D4RL](#d4rl)
    - [Run Training Reward Model](#run-training-reward-model)
      - [Preference Transformer (PT)](#preference-transformer-pt)
    - [Run IQL with Learned Reward Model](#run-iql-with-learned-reward-model)
      - [Preference Transformer (PT)](#preference-transformer-pt-1)
- [Acknowledgments](#acknowledgments)

## Installation

Follow the steps below to set up the environment and install the necessary dependencies.

1. **Create and activate a Conda environment:**
    ```bash
    conda create -y -n offline python=3.8
    conda activate offline
    ```

2. **Upgrade `pip`:**
    ```bash
    pip install --upgrade pip
    ```

3. **Install CUDA Toolkit and cuDNN:**
    ```bash
    conda install -y -c conda-forge cudatoolkit=11.1 cudnn=8.2.1
    ```

4. **Install Python dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

5. **Install D4RL:**
    ```bash
    cd d4rl
    pip install -e .
    cd ..
    ```

6. **Install JAX with CUDA support:**
    ```bash
    pip install "jax[cuda11_cudnn805]>=0.2.27" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
    ```

7. **Install additional packages:**
    ```bash
    pip install protobuf==3.20.1 gym<0.24.0 distrax==0.1.2 wandb transformers
    ```

## How to Run the Code

### D4RL

#### Run Training Reward Model

##### SPOT
```bash
CUDA_VISIBLE_DEVICES=0 python -m JaxPref.new_preference_reward_main \
    --use_human_label True \
    --comment {experiment_name} \
    --transformer.embd_dim 256 \
    --transformer.n_layer 1 \
    --transformer.n_head 4 \
    --env {D4RL_env_name} \
    --logging.output_dir './logs/pref_reward' \
    --batch_size 256 \
    --num_query {number_of_queries} \
    --query_len 100 \
    --n_epochs 10000 \
    --skip_flag 0 \
    --seed {seed} \
    --model_type PrefTransformer
```
### After running the above command, insert the path to subgoal_vae_{env_name}.pkl into the Learner class within learner.py.

### Run IQL with Learned Reward Model
```bash
CUDA_VISIBLE_DEVICES=0 python train_offline.py \
    --seq_len {sequence_length_in_reward_prediction} \
    --comment {experiment_name} \
    --eval_interval {5000:mujoco/100000:antmaze/50000:adroit} \
    --env_name {d4rl_env_name} \
    --config {configs/(mujoco|antmaze|adroit)_config.py} \
    --eval_episodes {100_for_ant, 10_otherwise} \
    --use_reward_model True \
    --model_type PrefTransformer \
    --ckpt_dir {reward_model_path} \
    --seed {seed}
```

## Acknowledgments
we implement our code based on [Preference Transformer](https://github.com/csmile-1006/PreferenceTransformer). 