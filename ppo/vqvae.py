import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizer(nn.Module):
    """
    A robust and numerically stable Vector Quantizer that uses Exponential
    Moving Average (EMA) updates for the codebook and includes a method
    for resetting "dead" codebook vectors.
    """

    def __init__(
        self,
        num_embeddings,
        embedding_dim,
        commitment_cost=0.25,
        decay=0.99,
        epsilon=1e-5,
    ):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.epsilon = epsilon

        # The codebook, which will be updated via EMA, not gradients.
        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.embedding.weight.data.normal_()

        # Buffers for EMA stats.
        self.register_buffer("_ema_cluster_size", torch.zeros(num_embeddings))
        self.register_buffer("_ema_w", self.embedding.weight.data.clone())

    def forward(self, inputs):
        # Shape inputs for distance calculation: (B, C, H, W) -> (B*H*W, C)
        inputs_permuted = (
            inputs.permute(0, 2, 3, 1).contiguous()
            if len(inputs.shape) == 4
            else inputs
        )
        input_shape = inputs_permuted.shape
        flat_input = inputs_permuted.view(-1, self.embedding_dim)

        # Calculate distances and find the closest codebook vectors
        distances = torch.cdist(flat_input, self.embedding.weight) ** 2
        encoding_indices = torch.argmin(distances, dim=1)

        encodings = F.one_hot(encoding_indices, self.num_embeddings).float()

        # --- EMA Codebook Update (only during training) ---
        if self.training:

            # Update EMA cluster size

            self._ema_cluster_size = self._ema_cluster_size * self.decay + (
                1 - self.decay
            ) * torch.sum(encodings, 0)

            # Update EMA weights

            dw = torch.matmul(encodings.t(), flat_input)

            # CORRECTION: Removed incorrect transpose on dw. Shape is (K, D).

            self._ema_w = self._ema_w * self.decay + (1 - self.decay) * dw

            n = torch.sum(self._ema_cluster_size)

            normalized_cluster_size = (
                (self._ema_cluster_size + self.epsilon)
                / (n + self.num_embeddings * self.epsilon)
                * n
            )

            dw_normalized = self._ema_w / normalized_cluster_size.unsqueeze(1)

            self.embedding.weight.data.copy_(dw_normalized)

        # --- Loss Calculation ---
        # The commitment loss trains the encoder. The .detach() prevents the codebook
        # from receiving gradients from this loss, as it's updated by EMA.
        loss = self.commitment_cost * F.mse_loss(
            flat_input, self.embedding(encoding_indices).detach()
        )

        # --- Quantization and Straight-Through Estimator ---
        quantized_flat = self.embedding(encoding_indices)
        quantized = quantized_flat.view(input_shape)

        quantized = inputs_permuted + (quantized - inputs_permuted).detach()
        if len(input_shape) == 4:
            quantized = quantized.permute(0, 3, 1, 2).contiguous()

        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return quantized, loss, perplexity, encodings

    def reset_dead_codes(self, batch_encoder_outputs):
        """
        Finds unused codebook vectors and re-initializes them by adding a small
        amount of noise to randomly selected *active* codebook vectors.
        """
        if not self.training:
            return

        with torch.no_grad():
            dead_codes_mask = self._ema_cluster_size < 1e-3
            active_codes_mask = ~dead_codes_mask
            num_dead = torch.sum(dead_codes_mask).item()
            num_active = self.num_embeddings - num_dead

            if num_dead == 0 or num_active == 0:
                return

            dead_indices = torch.where(dead_codes_mask)[0]
            active_indices = torch.where(active_codes_mask)[0]

            random_active_indices = active_indices[
                torch.randint(0, num_active, (num_dead,))
            ]

            replacement_vectors = self.embedding.weight.data[random_active_indices]
            noise = torch.randn_like(replacement_vectors) * 0.01
            replacement_vectors += noise

            self.embedding.weight.data[dead_indices] = replacement_vectors

            # Reset the EMA stats for these codes
            self._ema_w[dead_indices] = replacement_vectors
            self._ema_cluster_size[dead_indices] = 1.0  # Give it a small starting count

            print(f"Reset {num_dead} dead codebook vectors.")


class ResidualBlock(nn.Module):
    """
    A standard residual block with two convolutional layers.
    """

    def __init__(self, in_channels, out_channels, hidden_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=1, stride=1),
        )

    def forward(self, x):
        # The input is added to the output, creating the residual connection
        return x + self.block(x)


class ResidualStack(nn.Module):
    """
    A stack of residual blocks.
    """

    def __init__(self, in_channels, out_channels, hidden_channels, num_blocks):
        super().__init__()
        blocks = [
            ResidualBlock(in_channels, out_channels, hidden_channels)
            for _ in range(num_blocks)
        ]
        self.stack = nn.Sequential(*blocks)

    def forward(self, x):
        return self.stack(x)


class VQVAE_CNN(nn.Module):
    """
    A deeper VQ-VAE for 2D images that incorporates residual connections
    and maps the image to a GRID of embeddings.
    """

    def __init__(
        self,
        input_channels,
        num_embeddings,
        embedding_dim,
        hidden_dim=128,
        num_residual_blocks=2,
        commitment_cost=0.25,
        cluster_beta=0.7,
    ):
        super().__init__()

        # --- Deep Encoder with Residual Connections ---
        self.encoder = nn.Sequential(
            nn.Conv2d(
                input_channels, hidden_dim // 2, kernel_size=4, stride=2, padding=1
            ),
            nn.ReLU(),
            nn.Conv2d(hidden_dim // 2, hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            ResidualStack(hidden_dim, hidden_dim, hidden_dim // 2, num_residual_blocks),
            nn.Conv2d(hidden_dim, embedding_dim, kernel_size=1, stride=1),
        )

        self.vq_layer = VectorQuantizer(
            num_embeddings, embedding_dim, commitment_cost, cluster_beta
        )

        # --- Deep Decoder with Residual Connections ---
        self.decoder = nn.Sequential(
            nn.Conv2d(embedding_dim, hidden_dim, kernel_size=3, stride=1, padding=1),
            ResidualStack(hidden_dim, hidden_dim, hidden_dim // 2, num_residual_blocks),
            nn.ConvTranspose2d(
                hidden_dim, hidden_dim // 2, kernel_size=4, stride=2, padding=1
            ),
            nn.ReLU(),
            nn.ConvTranspose2d(
                hidden_dim // 2, input_channels, kernel_size=4, stride=2, padding=1
            ),
            nn.Sigmoid(),
        )
        self.training = True

    def forward(self, x):
        # The encoder produces a feature map of shape (B, embedding_dim, H/4, W/4)
        z_e = self.encoder(x)
        # The VQ layer quantizes each vector in the spatial grid
        quantized, vq_loss, perplexity, _ = self.vq_layer(z_e)
        # The decoder reconstructs the image from the quantized grid
        x_recon = self.decoder(quantized)
        return x_recon, vq_loss, perplexity

    # The encoder and decoder methods are implicitly defined by the main encoder/decoder attributes
    # but we can add them explicitly for clarity if needed.
    def encode(self, x):
        return self.encoder(x)

    def decode(self, z_q):
        return self.decoder(z_q)

    def reset_dead_codes(self, batch_encoder_outputs):

        self.vq_layer.reset_dead_codes(batch_encoder_outputs)


class VQVAE_MLP(nn.Module):
    """
    A VQ-VAE designed for 1D feature-based observations using MLPs.
    """

    def __init__(
        self, input_dim, hidden_dim, num_embeddings, embedding_dim, commitment_cost=0.25
    ):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim),  # Output matches embedding dimension
        )

        self.vq_layer = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)

        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),  # Reconstruct original input
        )

    def forward(self, x):
        z = self.encoder(x)
        quantized, vq_loss, perplexity, _ = self.vq_layer(z)
        x_recon = self.decoder(quantized)
        return x_recon, vq_loss, perplexity
