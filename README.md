# **Compositional Reinforcement Learning with Abstracted Representations**

This repository contains a research-level framework for exploring advanced reinforcement learning concepts. The core idea is to investigate how agents can learn to solve complex continuous control tasks by combining **learned state abstractions** (via Vector-Quantized VAEs) with **compositional action spaces** (via learnable action primitives).

The entire system is trained end-to-end using a robust Proximal Policy Optimization (PPO) implementation and is designed to be modular, allowing for easy experimentation with different agent architectures and observation types.

## **Core Features**

* **State Abstraction:** A VQ-VAE module can be enabled to compress high-dimensional state observations (either feature vectors or images) into a discrete set of learned "concepts".
* **Compositional Agents:** Includes AttentionPrototypeAgent that learn a set of reusable action "skills" and intelligently blend them to produce the final continuous action.
* **End-to-End Training:** A unified PPO training loop jointly optimizes all components, from the VQ-VAE's perception module to the agent's policy and the learned action primitives.

## **Installation**

1. **Clone the repository:**
   ```bash
   git clone https://github.com/TanmayAmbadkar/attention-rl.git
   cd attention-rl
   ```

2. **Create and activate a virtual environment** (recommended):

   ```bash
   python \-m venv venv
   source venv/bin/activate
   ```

3. Install the required dependencies:
   The project uses pre-commit hooks for code quality. After cloning and creating your virtual environment, install them.

   ```bash
   pip install \-r requirements.txt
   pre-commit install
   ```

## **How to Run**

The main entry point is main.py. You can configure different experiments using command-line arguments.

### **Example 1: Train a standard** ContinuousAgent **on features**

This is the baseline experiment using a standard PPO agent on a feature-based environment.

```bash
python main.py \--env\_id "Hopper-v5" \--agent\_type "continuous"
```

### **Example 2: Train the full** AttentionPrototypeAgent **on images**

This is the most advanced configuration. It uses rendered images as observations, a VQ-VAE to create a spatial grid of codes, and the attention-based agent to process it.

```bash
python main.py
    --env_id "Hopper-v5" \
    --agent_type "attention" \
    --use_image_obs True \
    --use_vqvae True \
    --num_prototypes 32 \
    --vq_embedding_dim 128 \
    --total_timesteps 2000000
```

## **Monitoring Training**

This project uses TensorBoard for logging. To launch it while training is in progress, run the following command in your terminal:

```bash
tensorboard \--logdir runs
```

Navigate to http://localhost:6006 in your browser. You can monitor key metrics such as charts/episodic\_return, losses/policy\_loss, and the VQ-VAE specific charts/vq\_perplexity.
