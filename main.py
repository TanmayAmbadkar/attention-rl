import os
import random
import time
from datetime import datetime
from functools import partial
from types import SimpleNamespace
from typing import Optional

import gymnasium as gym
import numpy as np
import torch
import torch.optim as optim
from func_to_script import script

# MODIFICATION: Assume a clean project structure for imports
from ppo import (
    PPO,
    VQVAE_CNN,
    VQVAE_MLP,
    AttentionPrototypeAgent,
    ContinuousAgent,
    PPOLogger,
)


def set_seed(seed, torch_deterministic=True):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = torch_deterministic


# MODIFICATION: These two new wrappers separate concerns cleanly.
class RenderObservation(gym.Wrapper):
    """
    A wrapper that replaces the standard observation with an RGB image rendering
    of the environment in its native (H, W, C) format.
    The wrapped environment MUST be created with `render_mode='rgb_array'`.
    """

    def __init__(self, env):
        super().__init__(env)
        self.env.reset()
        image_obs = self.env.render()
        if image_obs is None:
            raise ValueError(
                "render() returned None. Env must have `render_mode='rgb_array'`."
            )
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=image_obs.shape, dtype=np.uint8
        )

    def step(self, action):
        _, reward, terminated, truncated, info = self.env.step(action)
        return self.env.render(), reward, terminated, truncated, info

    def reset(self, **kwargs):
        _, info = self.env.reset(**kwargs)
        return self.env.render(), info


class PyTorchImageWrapper(gym.ObservationWrapper):
    """
    A wrapper that transforms an image observation for PyTorch compatibility.
    It transposes the image dimensions from (H, W, C) to (C, H, W).
    """

    def __init__(self, env):
        super().__init__(env)
        old_shape = self.observation_space.shape
        new_shape = (old_shape[2], old_shape[0], old_shape[1])
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=new_shape, dtype=np.uint8
        )

    def observation(self, observation):
        return np.transpose(observation, (2, 0, 1))


def make_continuous_env(
    env_id, idx, capture_video, run_name, gamma, use_image_obs=False
):
    """Creates a configured and wrapped continuous Gymnasium environment."""

    def create_configured_env():
        render_mode = (
            "rgb_array" if use_image_obs or (capture_video and idx == 0) else None
        )
        env = gym.make(env_id, render_mode=render_mode)

        if capture_video and idx == 0:
            env = gym.wrappers.RecordVideo(env, f"runs/{run_name}/videos")

        if use_image_obs:
            env = RenderObservation(env)
            env = gym.wrappers.ResizeObservation(env, (64, 64))
            env = PyTorchImageWrapper(env)
        else:
            env = gym.wrappers.FlattenObservation(env)
            # Normalization wrappers are typically used for feature-based obs
            env = gym.wrappers.NormalizeObservation(env)
            env = gym.wrappers.TransformObservation(
                env, lambda obs: np.clip(obs, -10, 10)
            )

        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env

    return create_configured_env


def create_envs(env_id, num_envs, capture_video, run_name, gamma, use_image_obs=False):
    """Creates a vectorized environment for continuous action spaces."""
    envs = gym.vector.SyncVectorEnv(
        [
            make_continuous_env(
                env_id, i, capture_video, run_name, gamma, use_image_obs
            )
            for i in range(num_envs)
        ]
    )
    return envs


def load_and_evaluate_model(
    run_name,
    env_id,
    agent_class,
    device,
    model_path,
    gamma,
    capture_video,
    # Add necessary params for instantiation
    use_image_obs,
    use_vqvae,
    vq_num_embeddings,
    vq_embedding_dim,
):
    """Loads a trained model and evaluates its performance."""
    print("--- Starting Evaluation ---")
    eval_episodes = 10
    eval_envs = create_envs(env_id, 1, capture_video, run_name, gamma, use_image_obs)

    # Instantiate models for evaluation
    vqvae, mock_env = None, eval_envs
    if use_vqvae:
        obs_shape = eval_envs.single_observation_space.shape
        if len(obs_shape) > 2:  # Image
            vqvae = VQVAE_CNN(obs_shape[0], vq_num_embeddings, vq_embedding_dim).to(
                device
            )
            # For attention agent, obs space is a grid
            with torch.no_grad():
                grid_shape = vqvae.encoder(torch.zeros(1, *obs_shape).to(device)).shape
            agent_obs_shape = grid_shape[1:]  # (C, H, W)
        else:  # Features
            vqvae = VQVAE_MLP(
                np.prod(obs_shape), 128, vq_num_embeddings, vq_embedding_dim
            ).to(device)
            agent_obs_shape = (vq_embedding_dim,)

        mock_obs_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=agent_obs_shape, dtype=np.float32
        )
        mock_env = SimpleNamespace(
            single_observation_space=mock_obs_space,
            single_action_space=eval_envs.single_action_space,
        )

    eval_agent = agent_class(mock_env).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    eval_agent.load_state_dict(checkpoint["agent_state_dict"])
    if use_vqvae:
        vqvae.load_state_dict(checkpoint["vqvae_state_dict"])
        vqvae.eval()
    eval_agent.eval()

    obs, _ = eval_envs.reset()
    episodic_returns = []
    while len(episodic_returns) < eval_episodes:
        with torch.no_grad():
            agent_obs = torch.Tensor(obs).to(device)
            if use_vqvae:
                agent_obs, _, _, _ = vqvae.vq_layer(vqvae.encoder(agent_obs))

            actions, _ = eval_agent.sample_action_and_compute_log_prob(agent_obs)

        obs, _, _, _, infos = eval_envs.step(actions.cpu().numpy())

        # MODIFICATION: Correctly handle vectorized `infos`
        if "episode" not in infos:
            continue
        info = infos["episode"]
        if not any(infos["_episode"]):
            return

        print(
            f"episodic_return={sum(info['r'][info['_r']])/(sum(info['_r']))}",
            flush=True,
        )
        episodic_returns += info["r"][info["_r"]].tolist()
    eval_envs.close()
    print("--- Evaluation Finished ---")


@script
def run_ppo(
    env_id: str = "Hopper-v4",
    agent_type: str = "attention",  # continuous, prototype, or attention
    use_image_obs: bool = True,
    use_vqvae: bool = True,
    num_prototypes: int = 32,
    vq_num_embeddings: int = 512,
    vq_embedding_dim: int = 128,
    vq_commitment_cost: float = 0.25,
    vq_temp_decay: float = 0.999,
    vq_loss_coefficient: float = 1.0,
    dissimilarity_loss_coefficient: float = 0.1,
    learning_rate: float = 0.0001,
    num_envs: int = 4,
    total_timesteps: int = 1000000,
    num_rollout_steps: int = 512,
    update_epochs: int = 20,
    num_minibatches: int = 32,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    surrogate_clip_threshold: float = 0.2,
    entropy_loss_coefficient: float = 0.0,
    value_function_loss_coefficient: float = 0.5,
    max_grad_norm: float = 0.5,
    target_kl: Optional[float] = None,
    anneal_lr: bool = True,
    normalize_advantages: bool = True,
    clip_value_function_loss: bool = True,
    seed: int = 1,
    torch_deterministic: bool = True,
    capture_video: bool = False,
    use_tensorboard: bool = True,
    save_model: bool = True,
):
    exp_name = "agent.rl_model"
    run_name = f"{env_id}__{agent_type}__{seed}__{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    set_seed(seed, torch_deterministic)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    envs = create_envs(env_id, num_envs, capture_video, run_name, gamma, use_image_obs)
    print(f"Created {num_envs} environments with ID: {env_id}")
    print(f"Environment observation space: {envs.single_observation_space}")
    print(f"Environment action space: {envs.single_action_space}")

    # --- VQ-VAE and Agent Setup ---
    vqvae, agent_class, mock_env = None, None, envs
    params_to_optimize = []

    if use_vqvae:
        print(f"Using VQ-VAE with embedding dim: {vq_embedding_dim}")
        obs_shape = envs.single_observation_space.shape
        if len(obs_shape) > 2:  # Image-like obs
            vqvae = VQVAE_CNN(
                obs_shape[0],
                vq_num_embeddings,
                vq_embedding_dim,
                vq_commitment_cost,
                temp_decay_rate=vq_temp_decay,
            ).to(device)
            # Calculate the spatial shape of the feature map for the mock env
            with torch.no_grad():
                grid_shape = vqvae.encoder(torch.zeros(1, *obs_shape).to(device)).shape
            agent_obs_shape = grid_shape[1:]  # (C, H, W)
        else:  # Feature-based obs
            vqvae = VQVAE_MLP(
                np.prod(obs_shape),
                128,
                vq_num_embeddings,
                vq_embedding_dim,
                vq_commitment_cost,
            ).to(device)
            agent_obs_shape = (vq_embedding_dim,)

        mock_obs_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=agent_obs_shape, dtype=np.float32
        )
        mock_env = SimpleNamespace(
            single_observation_space=mock_obs_space,
            single_action_space=envs.single_action_space,
        )
        params_to_optimize += list(vqvae.parameters())

    # --- Agent Selection ---
    if agent_type == "attention":
        if not use_vqvae or len(envs.single_observation_space.shape) <= 2:
            raise ValueError(
                "AttentionPrototypeAgent requires image observations and a VQ-VAE."
            )
        print("Using AttentionPrototypeAgent")
        agent_class = partial(AttentionPrototypeAgent, num_prototypes=num_prototypes)
    else:  # "continuous"
        print("Using standard ContinuousAgent.")
        agent_class = partial(ContinuousAgent)

    agent = agent_class(mock_env).to(device)
    params_to_optimize += list(agent.parameters())
    print(agent)

    optimizer = optim.Adam(params_to_optimize, lr=learning_rate, eps=1e-5)

    ppo = PPO(
        agent=agent,
        optimizer=optimizer,
        envs=envs,
        vqvae=vqvae,
        vq_embedding_dim=vq_embedding_dim if use_vqvae else None,
        vq_loss_coefficient=vq_loss_coefficient,
        dissimilarity_loss_coefficient=dissimilarity_loss_coefficient,
        learning_rate=learning_rate,
        num_rollout_steps=num_rollout_steps,
        num_envs=num_envs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        surrogate_clip_threshold=surrogate_clip_threshold,
        entropy_loss_coefficient=entropy_loss_coefficient,
        value_function_loss_coefficient=value_function_loss_coefficient,
        max_grad_norm=max_grad_norm,
        update_epochs=update_epochs,
        num_minibatches=num_minibatches,
        normalize_advantages=normalize_advantages,
        clip_value_function_loss=clip_value_function_loss,
        target_kl=target_kl,
        anneal_lr=anneal_lr,
        seed=seed,
        logger=PPOLogger(run_name, use_tensorboard),
    )

    trained_agent = ppo.learn(total_timesteps)

    if save_model:
        model_path = f"runs/{run_name}/{exp_name}.pt"
        os.makedirs(f"runs/{run_name}", exist_ok=True)
        checkpoint = {"agent_state_dict": trained_agent.state_dict()}
        if vqvae is not None:
            checkpoint["vqvae_state_dict"] = vqvae.state_dict()
        torch.save(checkpoint, model_path)
        print(f"Model saved to {model_path}")

        load_and_evaluate_model(
            run_name,
            env_id,
            agent_class,
            device,
            model_path,
            gamma,
            capture_video,
            use_image_obs,
            use_vqvae,
            vq_num_embeddings,
            vq_embedding_dim,
        )

    envs.close()


if __name__ == "__main__":
    run_ppo()
