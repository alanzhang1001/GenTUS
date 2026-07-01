import numpy as np
import torch
from vector_quantize_pytorch import ResidualVQ


RQVAE_ARTIFACT_VERSION = 1


def build_rqvae(
    dim: int,
    num_quantizers: int = 3,
    codebook_size: int = 64,
    commitment_weight: float = 0.25,
    kmeans_iters: int = 100,
) -> ResidualVQ:
    return ResidualVQ(
        dim=dim,
        num_quantizers=num_quantizers,
        codebook_size=codebook_size,
        commitment_weight=commitment_weight,
        kmeans_init=True,
        kmeans_iters=kmeans_iters,
        use_cosine_sim=True,
        orthogonal_reg_weight=10.0,
        threshold_ema_dead_code=1,
    )


def train(embeddings: np.ndarray, num_quantizers: int = 3, codebook_size: int = 64,
          epochs: int = 200, lr: float = 1e-3, commitment_weight: float = 0.25,
          batch_size: int = 256, device: str | None = None,
          kmeans_iters: int = 100, **kwargs) -> ResidualVQ:
    """Train an RQ-VAE-style residual quantizer with reconstruction and commitment loss."""
    if len(embeddings) == 0:
        raise ValueError("embeddings must not be empty")

    x = torch.tensor(embeddings, dtype=torch.float32)
    dim = x.shape[1]
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    rqvae = build_rqvae(
        dim=dim,
        num_quantizers=num_quantizers,
        codebook_size=codebook_size,
        commitment_weight=commitment_weight,
        kmeans_iters=kmeans_iters,
    )
    rqvae = rqvae.to(device)
    rqvae.train()
    optimizer = torch.optim.Adam(rqvae.parameters(), lr=lr)

    batch_size = min(batch_size, len(x))
    for epoch in range(epochs):
        perm = torch.randperm(len(x))
        epoch_loss = 0.0
        epoch_recon = 0.0
        epoch_commit = 0.0
        n_batches = 0

        for start in range(0, len(x), batch_size):
            idx = perm[start:start + batch_size]
            batch = x[idx].to(device)

            quantized, _, commit_loss = rqvae(batch)
            recon_loss = ((batch - quantized) ** 2).mean()
            commit_loss = commit_loss.mean()
            loss = recon_loss + commit_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_recon += recon_loss.item()
            epoch_commit += commit_loss.item()
            n_batches += 1

        if epoch % 20 == 0 or epoch == epochs - 1:
            print(
                f"  rqvae epoch {epoch + 1}/{epochs} "
                f"loss={epoch_loss / n_batches:.4f} "
                f"recon={epoch_recon / n_batches:.4f} "
                f"commit={epoch_commit / n_batches:.4f}"
            )

    rqvae = rqvae.cpu()
    return rqvae


def quantize(rqvae: ResidualVQ, embeddings: np.ndarray, batch_size: int = 1024,
             device: str | None = None) -> np.ndarray:
    """Returns integer indices of shape (N, num_quantizers)."""
    x = torch.tensor(embeddings, dtype=torch.float32)
    device = device or next(rqvae.parameters()).device
    rqvae = rqvae.to(device)
    rqvae.eval()
    all_indices = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            batch = x[start:start + batch_size].to(device)
            _, indices, _ = rqvae(batch)
            all_indices.append(indices.cpu())
    return torch.cat(all_indices, dim=0).numpy()


def save_rqvae_artifact(
    rqvae: ResidualVQ,
    path: str,
    *,
    dim: int,
    num_quantizers: int,
    codebook_size: int,
    commitment_weight: float,
    kmeans_iters: int = 100,
    extra_metadata: dict | None = None,
) -> None:
    payload = {
        "artifact_version": RQVAE_ARTIFACT_VERSION,
        "config": {
            "dim": dim,
            "num_quantizers": num_quantizers,
            "codebook_size": codebook_size,
            "commitment_weight": commitment_weight,
            "kmeans_iters": kmeans_iters,
        },
        "state_dict": rqvae.cpu().state_dict(),
        "metadata": extra_metadata or {},
    }
    torch.save(payload, path)


def load_rqvae_artifact(path: str, device: str | None = None) -> tuple[ResidualVQ, dict]:
    payload = torch.load(path, map_location=device or "cpu")
    config = payload["config"]
    rqvae = build_rqvae(
        dim=config["dim"],
        num_quantizers=config["num_quantizers"],
        codebook_size=config["codebook_size"],
        commitment_weight=config["commitment_weight"],
        kmeans_iters=config.get("kmeans_iters") or 100,
    )
    rqvae.load_state_dict(payload["state_dict"])
    rqvae = rqvae.to(device or "cpu")
    rqvae.eval()
    return rqvae, payload
