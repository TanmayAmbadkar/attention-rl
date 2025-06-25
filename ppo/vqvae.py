import torch
import torch.nn as nn
import torch.nn.functional as F


# The VectorQuantizer class remains unchanged as its logic is general.
class VectorQuantizer(nn.Module):
    """
    The core Vector Quantization layer with temperature-controlled exploration.
    This layer maps continuous embeddings to a discrete codebook vector, using
    Gumbel-Softmax for stochastic selection during training.
    """

    def __init__(
        self, num_embeddings, embedding_dim, commitment_cost, initial_temperature=1.0
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        self.temperature = initial_temperature

        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.embedding.weight.data.uniform_(
            -1.0 / self.num_embeddings, 1.0 / self.num_embeddings
        )

    def forward(self, inputs, is_training=True):
        # Convert inputs from (B, C, H, W) -> (B, H, W, C)
        inputs_permuted = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs_permuted.shape

        # Flatten input to (B*H*W, C) for distance calculation
        flat_input = inputs_permuted.view(-1, self.embedding_dim)

        distances = torch.cdist(flat_input, self.embedding.weight) ** 2

        if is_training:
            logits = -distances
            # print(logits)
            encodings = F.gumbel_softmax(logits / self.temperature, tau=1.0, hard=True)
        else:
            encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
            encodings = torch.zeros(
                encoding_indices.shape[0], self.num_embeddings, device=inputs.device
            )
            encodings.scatter_(1, encoding_indices, 1)

        # Get quantized vectors and reshape back to feature map shape
        quantized_flat = torch.matmul(encodings, self.embedding.weight)
        quantized = quantized_flat.view(input_shape)

        # Calculate losses
        codebook_loss = F.mse_loss(quantized, inputs_permuted.detach())
        commitment_loss = F.mse_loss(inputs_permuted, quantized.detach())
        loss = codebook_loss + self.commitment_cost * commitment_loss

        # Use straight-through estimator
        quantized = inputs_permuted + (quantized - inputs_permuted).detach()

        # Convert back to (B, C, H, W) for CNN layers
        quantized = quantized.permute(0, 3, 1, 2).contiguous()

        # Calculate perplexity
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return quantized, loss, perplexity, encodings

    def decay_temperature(self, decay_rate, min_temperature):
        self.temperature = max(self.temperature * decay_rate, min_temperature)


class VQVAE_CNN(nn.Module):
    """
    MODIFIED: A VQ-VAE for 2D images that maps the image to a GRID of embeddings.
    """

    def __init__(
        self,
        input_channels,
        num_embeddings,
        embedding_dim,
        commitment_cost=0.25,
        initial_temperature=10.0,
        temp_decay_rate=0.7,
        min_temperature=0.1,
    ):
        super().__init__()

        self.temp_decay_rate = temp_decay_rate
        self.min_temperature = min_temperature

        # --- CNN Encoder ---
        # The final number of channels in the encoder MUST match the embedding_dim
        # for the VQ layer to work correctly on the feature map.
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            # MODIFICATION: The final layer now outputs `embedding_dim` channels.
            nn.Conv2d(128, embedding_dim, kernel_size=3, stride=1, padding=1),
            # nn.ReLU(),
        )

        # MODIFICATION: Removed the aggregation layers (encoder_head and decoder_head)
        # The VQ layer will now operate on the entire feature map from the encoder.
        self.vq_layer = VectorQuantizer(
            num_embeddings, embedding_dim, commitment_cost, initial_temperature
        )

        # --- Decoder ---
        # The decoder now takes the quantized feature map directly.
        self.decoder_conv = nn.Sequential(
            # MODIFICATION: The first layer now takes `embedding_dim` as input channels.
            nn.ConvTranspose2d(embedding_dim, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, input_channels, kernel_size=4, stride=2, padding=1),
            # MODIFICATION: Removed final upsampling layer to match encoder's downsampling.
            nn.Sigmoid(),
        )

    def forward(self, x):
        # The forward pass is now simpler.
        z_e = self.encoder(x)
        quantized, vq_loss, perplexity, _ = self.vq_layer(
            z_e, is_training=self.training
        )
        x_recon = self.decoder(quantized)
        return x_recon, vq_loss, perplexity

    def encoder(self, x):
        # MODIFICATION: The encoder now just returns the feature map.
        return self.encoder_conv(x)

    def decoder(self, z_q):
        # MODIFICATION: The decoder now just takes the quantized feature map.
        return self.decoder_conv(z_q)

    def decay_temperature(self):
        """A convenience method to decay the temperature of the internal VQ layer."""
        self.vq_layer.decay_temperature(self.temp_decay_rate, self.min_temperature)
        print(f"Decayed VQ temperature to: {self.vq_layer.temperature:.4f}")


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
