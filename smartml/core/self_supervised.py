"""
smartml.core.self_supervised
============================
Self-supervised tabular representation learning using Masked Autoencoding and Denoising Autoencoder embeddings.
Learns high-dimensional latent representations and reconstructs corrupted tabular fields without ground-truth labels.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


@dataclass
class SelfSupervisedResult:
    embedding_dim: int
    n_features_in: int
    reconstruction_mse: float
    embeddings_df: pd.DataFrame
    model: MLPRegressor
    scaler: StandardScaler
    narrative: str
    steps: list[str] = field(default_factory=list)


class TabularSelfSupervisedLearner:
    """Masked / Denoising Autoencoder for self-supervised tabular representation learning.

    Corrupts tabular inputs with noise / masking and trains an autoencoder architecture
    to reconstruct the original table, generating rich dense embeddings for downstream tasks.

    Args:
        embedding_dim: Latent representation size (default: half of feature count, min 2).
        corruption_rate: Fraction of features corrupted/masked during training (default 0.2).
        random_state: Reproducibility seed.
    """

    def __init__(
        self,
        embedding_dim: Optional[int] = None,
        corruption_rate: float = 0.2,
        max_iter: int = 150,
        random_state: int = 42,
    ) -> None:
        self.embedding_dim = embedding_dim
        self.corruption_rate = corruption_rate
        self.max_iter = max_iter
        self.random_state = random_state

    def fit_transform(
        self,
        df: pd.DataFrame,
        target_col: Optional[str] = None,
    ) -> SelfSupervisedResult:
        steps = []
        work = df.copy()
        if target_col and target_col in work.columns:
            work = work.drop(columns=[target_col])

        num_df = work.select_dtypes(include=[np.number]).fillna(0)
        n_rows, n_cols = num_df.shape

        if n_cols < 2:
            raise ValueError("Self-supervised tabular representation learning requires at least 2 numeric features.")

        latent_dim = self.embedding_dim or max(2, n_cols // 2)

        scaler = StandardScaler()
        scaled_clean = scaler.fit_transform(num_df)

        steps.append(f"Standardized {n_cols} numeric features.")

        # Create corrupted / masked training inputs
        rng = np.random.RandomState(self.random_state)
        mask = rng.rand(*scaled_clean.shape) < self.corruption_rate
        corrupted_inputs = scaled_clean.copy()
        corrupted_inputs[mask] = 0.0  # Zero-mask corruption

        steps.append(f"Applied {self.corruption_rate * 100:.0f}% random masking corruption to inputs for reconstruction.")

        # Bottleneck autoencoder architecture (Input -> Latent -> Output)
        hidden_layer_sizes = (latent_dim * 2, latent_dim, latent_dim * 2)
        autoencoder = MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
            activation="relu",
            max_iter=self.max_iter,
            random_state=self.random_state,
        )
        autoencoder.fit(corrupted_inputs, scaled_clean)

        reconstructed = autoencoder.predict(scaled_clean)
        mse = float(np.mean((scaled_clean - reconstructed) ** 2))

        # Extract latent representations from first two hidden layer weights
        # Approximating feed-forward activations to the bottleneck latent layer
        W0 = autoencoder.coefs_[0]
        b0 = autoencoder.intercepts_[0]
        h0 = np.maximum(0, np.dot(scaled_clean, W0) + b0)

        W1 = autoencoder.coefs_[1]
        b1 = autoencoder.intercepts_[1]
        latent_embeddings = np.maximum(0, np.dot(h0, W1) + b1)

        embed_cols = [f"SSL_embed_{i + 1}" for i in range(latent_embeddings.shape[1])]
        embed_df = pd.DataFrame(latent_embeddings, columns=embed_cols, index=num_df.index)

        steps.append(f"Trained Denoising Autoencoder; extracted {embed_df.shape[1]}-dimensional latent embeddings.")

        narrative = (
            f"Self-Supervised Learning complete. Extracted {latent_embeddings.shape[1]}-dimensional embeddings "
            f"from {n_cols} features across {n_rows} rows. "
            f"Reconstruction MSE: {mse:.4f}."
        )

        return SelfSupervisedResult(
            embedding_dim=latent_embeddings.shape[1],
            n_features_in=n_cols,
            reconstruction_mse=round(mse, 4),
            embeddings_df=embed_df,
            model=autoencoder,
            scaler=scaler,
            narrative=narrative,
            steps=steps,
        )
