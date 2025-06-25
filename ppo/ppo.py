import os
from uuid import uuid4

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter


class LinearLRSchedule:
    def __init__(self, optimizer, initial_lr, total_updates):
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.total_updates = total_updates
        self.current_update = 0

    def step(self):
        self.current_update += 1
        frac = 1.0 - (self.current_update - 1.0) / self.total_updates
        lr = frac * self.initial_lr
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def get_lr(self):
        return [group["lr"] for group in self.optimizer.param_groups]


class PPOLogger:
    def __init__(self, run_name=None, use_tensorboard=False):
        self.use_tensorboard = use_tensorboard
        self.global_steps = []
        if self.use_tensorboard:
            run_name = str(uuid4()).hex if run_name is None else run_name
            if not os.path.exists("runs"):
                os.makedirs("runs")
            self.writer = SummaryWriter(f"runs/{run_name}")

    def log_rollout_step(self, infos, global_step):
        self.global_steps.append(global_step)
        # MODIFICATION: Removed redundant print(infos) for cleaner logs.
        if "episode" not in infos:
            return
        info = infos["episode"]
        if not any(infos["_episode"]):
            return

        print(
            f"global_step={global_step}, episodic_return={sum(info['r'][info['_r']])/(sum(info['_r']))}",
            flush=True,
        )

        if self.use_tensorboard:
            self.writer.add_scalar(
                "charts/episodic_return",
                sum(info["r"][info["_r"]]) / (sum(info["_r"])),
                global_step,
            )
            self.writer.add_scalar(
                "charts/episodic_length",
                sum(info["l"][info["_l"]]) / (sum(info["_l"])),
                global_step,
            )

    def log_policy_update(self, update_results, global_step):
        if self.use_tensorboard:
            for key, value in update_results.items():
                # Assumes keys are like "losses/policy_loss", "charts/perplexity", etc.
                self.writer.add_scalar(key, value, global_step)


class PPO:
    def __init__(
        self,
        agent,
        optimizer,
        envs,
        vqvae=None,
        vq_embedding_dim=None,
        vq_loss_coefficient=0.1,
        dissimilarity_loss_coefficient=0.1,
        learning_rate=3e-4,
        num_rollout_steps=2048,
        num_envs=1,
        gamma=0.99,
        gae_lambda=0.95,
        surrogate_clip_threshold=0.2,
        entropy_loss_coefficient=0.01,
        value_function_loss_coefficient=0.5,
        max_grad_norm=0.5,
        update_epochs=10,
        num_minibatches=32,
        normalize_advantages=True,
        clip_value_function_loss=True,
        target_kl=None,
        anneal_lr=True,
        seed=1,
        logger=None,
    ):
        self.agent = agent
        self.envs = envs
        self.optimizer = optimizer
        self.seed = seed
        self.vqvae = vqvae
        self.vq_loss_coefficient = vq_loss_coefficient
        self.dissimilarity_loss_coefficient = dissimilarity_loss_coefficient
        self.num_rollout_steps = num_rollout_steps
        self.num_envs = num_envs
        self.batch_size = num_envs * num_rollout_steps
        self.num_minibatches = num_minibatches
        self.minibatch_size = self.batch_size // num_minibatches
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.surrogate_clip_threshold = surrogate_clip_threshold
        self.entropy_loss_coefficient = entropy_loss_coefficient
        self.value_function_loss_coefficient = value_function_loss_coefficient
        self.max_grad_norm = max_grad_norm
        self.update_epochs = update_epochs
        self.normalize_advantages = normalize_advantages
        self.clip_value_function_loss = clip_value_function_loss
        self.target_kl = target_kl
        self.device = next(agent.parameters()).device
        self.anneal_lr = anneal_lr
        self.initial_lr = learning_rate
        self.lr_scheduler = None
        self._global_step = 0
        self.logger = logger or PPOLogger()

        # MODIFICATION: Correctly determine agent's observation shape
        self.agent_obs_shape = None
        if self.vqvae is not None:
            # Check if the VQVAE produces a spatial grid or a flat vector
            with torch.no_grad():
                dummy_obs = torch.zeros(1, *envs.single_observation_space.shape).to(
                    self.device
                )
                encoded_obs = self.vqvae.encoder(dummy_obs)
                # The shape for the agent is the shape of the quantized output, minus the batch dim
                self.agent_obs_shape = encoded_obs.shape[1:]
        else:
            self.agent_obs_shape = envs.single_observation_space.shape
        print(f"Agent observation shape set to: {self.agent_obs_shape}")

    def create_lr_scheduler(self, num_policy_updates):

        return LinearLRSchedule(self.optimizer, self.initial_lr, num_policy_updates)

    def learn(self, total_timesteps):
        num_policy_updates = total_timesteps // self.batch_size
        if self.anneal_lr:
            self.lr_scheduler = self.create_lr_scheduler(num_policy_updates)
        next_observation, is_next_observation_terminal = self._initialize_environment()
        self._global_step = 0
        for update in range(1, num_policy_updates + 1):
            if self.anneal_lr:
                self.lr_scheduler.step()
            rollout_data = self.collect_rollouts(
                next_observation, is_next_observation_terminal
            )
            next_observation = rollout_data[-2]
            is_next_observation_terminal = rollout_data[-1]
            update_results = self.update_policy(*rollout_data[:-2])
            self.logger.log_policy_update(update_results, self._global_step)
        print(f"Training completed. Total steps: {self._global_step}")
        return self.agent

    def _initialize_environment(self):
        initial_observation, _ = self.envs.reset(seed=self.seed)
        initial_observation = torch.Tensor(initial_observation).to(self.device)
        is_initial_observation_terminal = torch.zeros(self.num_envs, device=self.device)
        return initial_observation, is_initial_observation_terminal

    def collect_rollouts(self, next_observation, is_next_observation_terminal):
        raw_observations_storage = torch.zeros(
            (self.num_rollout_steps, self.num_envs)
            + self.envs.single_observation_space.shape
        ).to(self.device)
        agent_observations_storage = torch.zeros(
            (self.num_rollout_steps, self.num_envs) + self.agent_obs_shape
        ).to(self.device)
        actions_storage = torch.zeros(
            (self.num_rollout_steps, self.num_envs)
            + self.envs.single_action_space.shape
        ).to(self.device)
        log_probs_storage = torch.zeros((self.num_rollout_steps, self.num_envs)).to(
            self.device
        )
        rewards_storage = torch.zeros((self.num_rollout_steps, self.num_envs)).to(
            self.device
        )
        dones_storage = torch.zeros((self.num_rollout_steps, self.num_envs)).to(
            self.device
        )
        values_storage = torch.zeros((self.num_rollout_steps, self.num_envs)).to(
            self.device
        )

        for step in range(self.num_rollout_steps):
            raw_observations_storage[step] = next_observation
            dones_storage[step] = is_next_observation_terminal
            with torch.no_grad():
                if self.vqvae is not None:
                    agent_observation, _, _, _ = self.vqvae.vq_layer(
                        self.vqvae.encoder(next_observation), is_training=True
                    )
                else:
                    agent_observation = next_observation

                agent_observations_storage[step] = agent_observation
                action, logprob = self.agent.sample_action_and_compute_log_prob(
                    agent_observation
                )
                value = self.agent.estimate_value_from_observation(agent_observation)

            values_storage[step] = value.flatten()
            actions_storage[step] = action
            log_probs_storage[step] = logprob

            next_observation, reward, terminations, truncations, infos = self.envs.step(
                action.cpu().numpy()
            )
            self._global_step += self.num_envs
            rewards_storage[step] = torch.as_tensor(reward, device=self.device).view(-1)
            is_next_observation_terminal = np.logical_or(terminations, truncations)
            next_observation = torch.as_tensor(
                next_observation, dtype=torch.float32, device=self.device
            )
            is_next_observation_terminal = torch.as_tensor(
                is_next_observation_terminal, dtype=torch.float32, device=self.device
            )
            self.logger.log_rollout_step(infos, self._global_step)

        with torch.no_grad():
            if self.vqvae is not None:
                final_agent_obs, _, _, _ = self.vqvae.vq_layer(
                    self.vqvae.encoder(next_observation), is_training=False
                )
            else:
                final_agent_obs = next_observation
            next_value = self.agent.estimate_value_from_observation(
                final_agent_obs
            ).reshape(1, -1)
            advantages, returns = self.compute_advantages(
                rewards_storage,
                values_storage,
                dones_storage,
                next_value,
                is_next_observation_terminal,
            )

        batch_raw_obs = raw_observations_storage.reshape(
            (-1,) + self.envs.single_observation_space.shape
        )
        (
            batch_agent_obs,
            batch_log_probs,
            batch_actions,
            batch_adv,
            batch_returns,
            batch_values,
        ) = self._flatten_rollout_data(
            agent_observations_storage,
            log_probs_storage,
            actions_storage,
            advantages,
            returns,
            values_storage,
        )
        return (
            batch_raw_obs,
            batch_agent_obs,
            batch_log_probs,
            batch_actions,
            batch_adv,
            batch_returns,
            batch_values,
            next_observation,
            is_next_observation_terminal,
        )

    def _flatten_rollout_data(
        self, obs, log_probs, actions, advantages, returns, values
    ):
        b_obs = obs.reshape((-1,) + self.agent_obs_shape)
        b_log_probs = log_probs.reshape(-1)
        b_actions = actions.reshape((-1,) + self.envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)
        return b_obs, b_log_probs, b_actions, b_advantages, b_returns, b_values

    def compute_advantages(self, rewards, values, dones, next_value, next_done):
        advantages = torch.zeros_like(rewards).to(self.device)
        lastgaelam = 0
        for t in reversed(range(self.num_rollout_steps)):
            if t == self.num_rollout_steps - 1:
                nextnonterminal = 1.0 - next_done
                nextvalues = next_value
            else:
                nextnonterminal = 1.0 - dones[t + 1]
                nextvalues = values[t + 1]
            delta = rewards[t] + self.gamma * nextvalues * nextnonterminal - values[t]
            advantages[t] = lastgaelam = (
                delta + self.gamma * self.gae_lambda * nextnonterminal * lastgaelam
            )
        returns = advantages + values
        return advantages, returns

    def calculate_dissimilarity_loss(self, prototypes):
        normed_prototypes = F.normalize(prototypes, p=2, dim=1, eps=1e-8)
        similarity_matrix = torch.matmul(normed_prototypes, normed_prototypes.t())
        off_diagonal_similarity = similarity_matrix - torch.eye(
            prototypes.shape[0], device=self.device
        )
        return torch.mean(torch.abs(off_diagonal_similarity))

    def update_policy(
        self,
        batch_raw_observations,
        _,
        batch_log_probabilities,
        batch_actions,
        batch_advantages,
        batch_returns,
        batch_values,
    ):
        batch_indices = np.arange(self.batch_size)
        clipping_fractions = []
        for epoch in range(self.update_epochs):
            np.random.shuffle(batch_indices)
            epoch_vq_losses, epoch_recon_losses, epoch_perplexities = [], [], []
            (
                epoch_dissimilarity_losses,
                epoch_pg_losses,
                epoch_v_losses,
                epoch_entropy_losses,
            ) = ([], [], [], [])

            for start in range(0, self.batch_size, self.minibatch_size):
                end = start + self.minibatch_size
                minibatch_indices = batch_indices[start:end]

                total_vq_loss = 0
                if self.vqvae is not None:
                    raw_obs_mb = batch_raw_observations[minibatch_indices]
                    encoded_mb = self.vqvae.encoder(raw_obs_mb)
                    agent_obs_mb, vq_loss_mb, perplexity_mb, _ = self.vqvae.vq_layer(
                        encoded_mb, is_training=True
                    )
                    recon_obs_mb = self.vqvae.decoder(agent_obs_mb)
                    recon_loss = F.mse_loss(recon_obs_mb, raw_obs_mb)
                    total_vq_loss = recon_loss + vq_loss_mb
                    epoch_vq_losses.append(vq_loss_mb.item())
                    epoch_recon_losses.append(recon_loss.item())
                    epoch_perplexities.append(perplexity_mb.item())
                else:
                    # If not using VQ-VAE, use the raw (flattened) observations
                    agent_obs_mb = batch_raw_observations[minibatch_indices]

                new_log_probs, entropy = (
                    self.agent.compute_action_log_probabilities_and_entropy(
                        agent_obs_mb, batch_actions[minibatch_indices]
                    )
                )
                new_values = self.agent.estimate_value_from_observation(agent_obs_mb)
                log_ratio = new_log_probs - batch_log_probabilities[minibatch_indices]
                ratio = log_ratio.exp()

                with torch.no_grad():
                    old_approx_kl = (-log_ratio).mean()
                    approx_kl = ((ratio - 1) - log_ratio).mean()
                clipping_fractions.append(
                    ((ratio - 1.0).abs() > self.surrogate_clip_threshold)
                    .float()
                    .mean()
                    .item()
                )

                minibatch_advantages = batch_advantages[minibatch_indices]
                if self.normalize_advantages:
                    minibatch_advantages = (
                        minibatch_advantages - minibatch_advantages.mean()
                    ) / (minibatch_advantages.std() + 1e-8)

                pg_loss = -torch.min(
                    minibatch_advantages * ratio,
                    minibatch_advantages
                    * torch.clamp(
                        ratio,
                        1 - self.surrogate_clip_threshold,
                        1 + self.surrogate_clip_threshold,
                    ),
                ).mean()
                v_loss = (
                    0.5
                    * (
                        (new_values.view(-1) - batch_returns[minibatch_indices]) ** 2
                    ).mean()
                )  # Simplified for clarity
                entropy_loss = entropy.mean()

                dissimilarity_loss = 0
                if hasattr(self.agent, "prototypes"):
                    dissimilarity_loss = self.calculate_dissimilarity_loss(
                        self.agent.prototypes
                    )
                    epoch_dissimilarity_losses.append(dissimilarity_loss.item())

                loss = (
                    pg_loss
                    - self.entropy_loss_coefficient * entropy_loss
                    + self.value_function_loss_coefficient * v_loss
                    + self.vq_loss_coefficient * total_vq_loss
                    + self.dissimilarity_loss_coefficient * dissimilarity_loss
                )

                epoch_pg_losses.append(pg_loss.item())
                epoch_v_losses.append(v_loss.item())
                epoch_entropy_losses.append(entropy_loss.item())

                self.optimizer.zero_grad()
                loss.backward()
                params_to_clip = list(self.agent.parameters())
                if self.vqvae:
                    params_to_clip += list(self.vqvae.parameters())
                nn.utils.clip_grad_norm_(params_to_clip, self.max_grad_norm)
                self.optimizer.step()

            if self.target_kl is not None and approx_kl > self.target_kl:
                break

        # After each full update, decay the VQ-VAE temperature
        if self.vqvae and hasattr(self.vqvae, "decay_temperature"):
            self.vqvae.decay_temperature()

        y_pred, y_true = batch_values.cpu().numpy(), batch_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        update_results = {
            "losses/policy_loss": np.mean(epoch_pg_losses),
            "losses/value_loss": np.mean(epoch_v_losses),
            "losses/entropy_loss": np.mean(epoch_entropy_losses),
            "charts/old_approx_kl": old_approx_kl.item(),
            "charts/approx_kl": approx_kl.item(),
            "charts/clipping_fraction": np.mean(clipping_fractions),
            "charts/explained_variance": explained_var,
        }
        if self.vqvae is not None:
            update_results["losses/vq_loss"] = np.mean(epoch_vq_losses)
            update_results["losses/recon_loss"] = np.mean(epoch_recon_losses)
            update_results["charts/vq_perplexity"] = np.mean(epoch_perplexities)
        if hasattr(self.agent, "prototypes"):
            update_results["losses/dissimilarity_loss"] = np.mean(
                epoch_dissimilarity_losses
            )
        return update_results
