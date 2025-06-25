from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """
    Initialize a linear or convolutional layer with orthogonal initialization for weights
    and a constant bias.
    """
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class BaseAgent(nn.Module, ABC):
    """
    Abstract Base Class for reinforcement learning agents.
    It defines the common interface that all agents must implement.
    """

    @abstractmethod
    def estimate_value_from_observation(self, observation):
        pass

    @abstractmethod
    def get_action_distribution(self, observation):
        pass

    @abstractmethod
    def sample_action_and_compute_log_prob(self, observations):
        pass

    @abstractmethod
    def compute_action_log_probabilities_and_entropy(self, observations, actions):
        pass


class ContinuousAgent(BaseAgent):
    """
    A standard agent for continuous action spaces, using a Normal distribution.
    This version is adapted to handle both flat and image-based observations.
    """

    def __init__(self, envs, rpo_alpha=None):
        super().__init__()
        self.rpo_alpha = rpo_alpha
        obs_shape = envs.single_observation_space.shape

        # --- Critic Network ---
        # Handles both image (CNN) and feature (MLP) observations.
        if len(obs_shape) > 1:  # Image-based
            self.critic_body = nn.Sequential(
                layer_init(nn.Conv2d(obs_shape[0], 32, kernel_size=8, stride=4)),
                nn.ReLU(),
                layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2)),
                nn.ReLU(),
                layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1)),
                nn.ReLU(),
                nn.Flatten(),
            )
            # Calculate the flattened size after the CNN
            with torch.no_grad():
                dummy_input = torch.zeros(1, *obs_shape)
                flattened_size = self.critic_body(dummy_input).shape[1]

            self.critic_head = layer_init(nn.Linear(flattened_size, 1), std=1.0)
        else:  # Feature-based
            self.critic_body = nn.Sequential(
                layer_init(nn.Linear(np.prod(obs_shape), 64)),
                nn.Tanh(),
                layer_init(nn.Linear(64, 64)),
                nn.Tanh(),
            )
            self.critic_head = layer_init(nn.Linear(64, 1), std=1.0)

        # --- Actor Network ---
        # Also handles both image and feature observations.
        if len(obs_shape) > 1:  # Image-based
            self.actor_body = nn.Sequential(
                layer_init(nn.Conv2d(obs_shape[0], 32, kernel_size=8, stride=4)),
                nn.ReLU(),
                layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2)),
                nn.ReLU(),
                layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1)),
                nn.ReLU(),
                nn.Flatten(),
            )
            with torch.no_grad():
                dummy_input = torch.zeros(1, *obs_shape)
                flattened_size = self.actor_body(dummy_input).shape[1]
            self.actor_mean_head = layer_init(
                nn.Linear(flattened_size, np.prod(envs.single_action_space.shape)),
                std=0.01,
            )
        else:  # Feature-based
            self.actor_body = nn.Sequential(
                layer_init(nn.Linear(np.prod(obs_shape), 64)),
                nn.Tanh(),
                layer_init(nn.Linear(64, 64)),
                nn.Tanh(),
            )
            self.actor_mean_head = layer_init(
                nn.Linear(64, np.prod(envs.single_action_space.shape)), std=0.01
            )

        self.actor_logstd = nn.Parameter(
            torch.zeros(1, np.prod(envs.single_action_space.shape))
        )

    def estimate_value_from_observation(self, observation):
        features = self.critic_body(observation)
        return self.critic_head(features)

    def get_action_distribution(self, observation):
        features = self.actor_body(observation)
        action_mean = self.actor_mean_head(features)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        return Normal(action_mean, action_std)

    def sample_action_and_compute_log_prob(self, observations):
        action_dist = self.get_action_distribution(observations)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action).sum(1)
        return action, log_prob

    def compute_action_log_probabilities_and_entropy(self, observations, actions):
        action_dist = self.get_action_distribution(observations)
        log_prob = action_dist.log_prob(actions).sum(1)
        entropy = action_dist.entropy().sum(1)
        return log_prob, entropy


class AttentionPrototypeAgent(ContinuousAgent):
    """
    An agent that uses an attention mechanism over a spatial grid of embeddings
    to produce a blended action from a set of learnable prototypes.
    """

    def __init__(self, envs, num_prototypes=10, rpo_alpha=None):
        super(
            ContinuousAgent, self
        ).__init__()  # Skip parent's __init__ to redefine networks
        self.rpo_alpha = rpo_alpha
        obs_shape = envs.single_observation_space.shape  # e.g., (embedding_dim, H, W)
        action_dim = np.prod(envs.single_action_space.shape)

        if len(obs_shape) < 3:
            raise ValueError(
                "AttentionPrototypeAgent requires a grid-like observation (e.g., from a CNN VQ-VAE)."
            )

        embedding_dim, H, W = obs_shape
        num_spatial_locations = H * W

        # --- Critic Network ---
        # CNN to process the spatial feature grid and output a single value
        self.critic = nn.Sequential(
            layer_init(nn.Conv2d(embedding_dim, 64, kernel_size=3, padding=1)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 32, kernel_size=3, padding=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(32 * H * W, 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, 1), std=1.0),
        )

        # --- Action Prototypes ---
        self.prototypes = nn.Parameter(torch.randn(num_prototypes, action_dim))
        torch.nn.init.orthogonal_(self.prototypes)

        # --- Actor Network (Shared Body + Two Heads) ---
        self.actor_body = nn.Sequential(
            layer_init(nn.Conv2d(embedding_dim, 64, kernel_size=3, padding=1)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, kernel_size=3, padding=1)),
            nn.ReLU(),
        )

        # Head 1: Local Action Gater
        # For each spatial location, outputs logits over the prototypes.
        self.local_action_head = layer_init(
            nn.Conv2d(64, num_prototypes, kernel_size=1), std=0.01
        )

        # Head 2: Spatial Attention Gater
        # Outputs a single attention score for each spatial location.
        self.spatial_attention_head = nn.Sequential(
            layer_init(nn.Conv2d(64, 1, kernel_size=1), std=0.01), nn.Flatten()
        )

        self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))

    def estimate_value_from_observation(self, observation):
        return self.critic(observation)

    def get_action_distribution(self, observation):
        # observation shape: (B, C, H, W) where C is embedding_dim
        B, C, H, W = observation.shape
        num_spatial_locations = H * W
        action_dim = self.prototypes.shape[1]

        # 1. Pass observation through shared actor body
        shared_features = self.actor_body(observation)  # (B, 64, H, W)

        # 2. Get local action logits and weights
        local_action_logits = self.local_action_head(
            shared_features
        )  # (B, num_prototypes, H, W)
        local_action_weights = torch.softmax(
            local_action_logits, dim=1
        )  # Softmax over prototypes

        # 3. Get spatial attention logits and weights
        spatial_attention_logits = self.spatial_attention_head(
            shared_features
        )  # (B, H*W)
        spatial_attention_weights = torch.softmax(
            spatial_attention_logits, dim=1
        )  # (B, H*W)

        # 4. Blend prototypes locally for each spatial location
        # Reshape for matrix multiplication:
        # (B, H, W, num_prototypes) @ (num_prototypes, action_dim) -> (B, H, W, action_dim)
        local_weights_permuted = local_action_weights.permute(0, 2, 3, 1)
        local_blended_actions = (
            local_weights_permuted @ self.prototypes
        )  # (B, H, W, action_dim)

        # 5. Apply spatial attention to get the final blended action mean
        # Flatten local actions to match attention weights
        local_blended_flat = local_blended_actions.view(
            B, num_spatial_locations, action_dim
        )
        # Expand attention weights for broadcasting
        attention_expanded = spatial_attention_weights.unsqueeze(-1)  # (B, H*W, 1)

        # Perform weighted sum
        final_blended_mean = (local_blended_flat * attention_expanded).sum(
            dim=1
        )  # (B, action_dim)

        # 6. Create the final Normal distribution for exploration
        action_logstd = self.actor_logstd.expand_as(final_blended_mean)
        action_std = torch.exp(action_logstd)
        action_dist = Normal(final_blended_mean, action_std)

        return action_dist
