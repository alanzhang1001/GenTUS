from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from vector_quantize_pytorch import ResidualVQ


TASK_AWARE_RQVAE_ARTIFACT_VERSION = 1


def build_residual_vq(
    dim: int,
    num_quantizers: int = 4,
    codebook_size: int = 256,
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


def _parse_hidden_dims(spec: str | list[int] | tuple[int, ...] | None) -> list[int]:
    if spec is None:
        return [512, 256]
    if isinstance(spec, (list, tuple)):
        return [int(v) for v in spec if int(v) > 0]
    return [int(v.strip()) for v in str(spec).split(",") if v.strip()]


def _build_mlp(dims: list[int], dropout: float = 0.0, final_activation: bool = False) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(len(dims) - 1):
        in_dim = dims[i]
        out_dim = dims[i + 1]
        layers.append(nn.Linear(in_dim, out_dim))
        is_last = i == len(dims) - 2
        if not is_last or final_activation:
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


@dataclass
class TaskAwareBatchOutput:
    z: torch.Tensor
    quantized: torch.Tensor
    indices: torch.Tensor
    reconstructed: torch.Tensor
    rq_loss: torch.Tensor


class TaskAwareRQVAE(nn.Module):
    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 128,
        hidden_dims: list[int] | None = None,
        num_quantizers: int = 4,
        codebook_size: int = 256,
        commitment_weight: float = 0.25,
        dropout: float = 0.0,
        l2_normalize_latent: bool = True,
        kmeans_iters: int = 100,
    ):
        super().__init__()
        hidden_dims = _parse_hidden_dims(hidden_dims)
        enc_dims = [input_dim, *hidden_dims, latent_dim]
        dec_dims = [latent_dim, *reversed(hidden_dims), input_dim]
        self.encoder = _build_mlp(enc_dims, dropout=dropout, final_activation=False)
        self.decoder = _build_mlp(dec_dims, dropout=dropout, final_activation=False)
        self.rqvae = build_residual_vq(
            dim=latent_dim,
            num_quantizers=num_quantizers,
            codebook_size=codebook_size,
            commitment_weight=commitment_weight,
            kmeans_iters=kmeans_iters,
        )
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.num_quantizers = num_quantizers
        self.codebook_size = codebook_size
        self.commitment_weight = commitment_weight
        self.dropout = dropout
        self.l2_normalize_latent = l2_normalize_latent
        self.kmeans_iters = kmeans_iters

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        if self.l2_normalize_latent:
            z = F.normalize(z, dim=-1)
        return z

    def forward(self, x: torch.Tensor) -> TaskAwareBatchOutput:
        z = self.encode(x)
        quantized, indices, rq_loss = self.rqvae(z)
        reconstructed = self.decoder(quantized)
        return TaskAwareBatchOutput(
            z=z,
            quantized=quantized,
            indices=indices,
            reconstructed=reconstructed,
            rq_loss=rq_loss,
        )


def build_positive_adjacency(num_items: int, positive_pairs: list[tuple[int, int]]) -> list[set[int]]:
    adjacency = [set() for _ in range(num_items)]
    for i, j in positive_pairs:
        if i == j:
            continue
        adjacency[i].add(j)
        adjacency[j].add(i)
    return adjacency


def build_potential_positive_adjacency(num_items: int,
                                       positive_adjacency: list[set[int]]) -> list[set[int]]:
    """Transitive closure (within each connected component) of the positive
    graph. Tables linked through a shared neighbor are likely unionable too,
    so they should be excluded from the in-batch negatives (false negatives)."""
    seen = [False] * num_items
    comp_of = [-1] * num_items
    comps: list[list[int]] = []
    for s in range(num_items):
        if seen[s] or not positive_adjacency[s]:
            continue
        stack = [s]; seen[s] = True; comp = []
        while stack:
            u = stack.pop(); comp.append(u); comp_of[u] = len(comps)
            for v in positive_adjacency[u]:
                if not seen[v]:
                    seen[v] = True; stack.append(v)
        comps.append(comp)
    potential = [set() for _ in range(num_items)]
    for comp in comps:
        cset = set(comp)
        for u in comp:
            potential[u] = cset - {u}
    return potential


def supervised_contrastive_loss(
    z: torch.Tensor,
    batch_indices: torch.Tensor,
    positive_adjacency: list[set[int]],
    temperature: float = 0.1,
    exclude_adjacency: list[set[int]] | None = None,
) -> tuple[torch.Tensor, int]:
    if z.size(0) <= 1:
        return z.new_tensor(0.0), 0

    z = F.normalize(z, dim=-1)
    sims = torch.matmul(z, z.T) / temperature
    eye = torch.eye(z.size(0), dtype=torch.bool, device=z.device)
    sims = sims.masked_fill(eye, float("-inf"))

    losses = []
    pair_count = 0
    batch_id_list = batch_indices.detach().cpu().tolist()
    for local_i, global_i in enumerate(batch_id_list):
        positive_globals = positive_adjacency[global_i]
        positive_locals = [
            local_j
            for local_j, global_j in enumerate(batch_id_list)
            if local_j != local_i and global_j in positive_globals
        ]
        if not positive_locals:
            continue
        row = sims[local_i]
        # Mask false negatives: potential positives that are not labeled
        # positives of this anchor are removed from the denominator so they
        # are neither pulled nor pushed.
        if exclude_adjacency is not None:
            excl = exclude_adjacency[global_i]
            if excl:
                fn_locals = [
                    local_j
                    for local_j, global_j in enumerate(batch_id_list)
                    if local_j != local_i
                    and global_j in excl
                    and global_j not in positive_globals
                ]
                if fn_locals:
                    row = row.clone()
                    row[fn_locals] = float("-inf")
        log_denom = torch.logsumexp(row, dim=0)
        pos_logits = row[positive_locals]
        losses.append(-(pos_logits - log_denom).mean())
        pair_count += len(positive_locals)

    if not losses:
        return z.new_tensor(0.0), 0
    return torch.stack(losses).mean(), pair_count


def build_training_batches(
    num_items: int,
    batch_size: int,
    positive_adjacency: list[set[int]],
    rng: np.random.Generator,
) -> list[np.ndarray]:
    if num_items == 0:
        return []

    unused = set(range(num_items))
    batches = []
    while unused:
        anchor = unused.pop()
        batch = [anchor]

        positive_candidates = [idx for idx in positive_adjacency[anchor] if idx in unused]
        rng.shuffle(positive_candidates)
        for idx in positive_candidates:
            if len(batch) >= batch_size:
                break
            batch.append(idx)
            unused.remove(idx)

        if len(batch) < batch_size:
            remaining = list(unused)
            rng.shuffle(remaining)
            for idx in remaining[: batch_size - len(batch)]:
                batch.append(idx)
                unused.remove(idx)

        batches.append(np.asarray(batch, dtype=np.int64))

    return batches


def train_task_aware_rqvae(
    embeddings: np.ndarray,
    positive_pairs: list[tuple[int, int]],
    num_quantizers: int = 4,
    codebook_size: int = 256,
    epochs: int = 200,
    lr: float = 1e-3,
    commitment_weight: float = 0.25,
    batch_size: int = 256,
    latent_dim: int = 128,
    hidden_dims: str | list[int] | tuple[int, ...] | None = None,
    union_loss_weight: float = 0.1,
    temperature: float = 0.1,
    dropout: float = 0.0,
    device: str | None = None,
    seed: int = 42,
    kmeans_iters: int = 100,
    mask_false_negatives: bool = False,
    extra_exclude_adjacency: list[set[int]] | None = None,
) -> tuple[TaskAwareRQVAE, dict]:
    if len(embeddings) == 0:
        raise ValueError("embeddings must not be empty")

    x = torch.tensor(embeddings, dtype=torch.float32)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = TaskAwareRQVAE(
        input_dim=x.shape[1],
        latent_dim=latent_dim,
        hidden_dims=_parse_hidden_dims(hidden_dims),
        num_quantizers=num_quantizers,
        codebook_size=codebook_size,
        commitment_weight=commitment_weight,
        dropout=dropout,
        kmeans_iters=kmeans_iters,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    adjacency = build_positive_adjacency(len(embeddings), positive_pairs)
    exclude_adjacency = None
    if mask_false_negatives or extra_exclude_adjacency is not None:
        exclude_adjacency = [set() for _ in range(len(embeddings))]
        if mask_false_negatives:
            pot = build_potential_positive_adjacency(len(embeddings), adjacency)
            for i in range(len(embeddings)):
                exclude_adjacency[i] |= pot[i]
        if extra_exclude_adjacency is not None:
            for i in range(min(len(embeddings), len(extra_exclude_adjacency))):
                exclude_adjacency[i] |= extra_exclude_adjacency[i]
    rng = np.random.default_rng(seed)
    batch_size = min(batch_size, len(x))
    history = []

    for epoch in range(epochs):
        model.train()
        batches = build_training_batches(len(x), batch_size, adjacency, rng)
        epoch_total = 0.0
        epoch_recon = 0.0
        epoch_rq = 0.0
        epoch_union = 0.0
        epoch_union_pairs = 0

        for batch_ids_np in batches:
            batch_ids = torch.tensor(batch_ids_np, dtype=torch.long)
            batch = x[batch_ids].to(device)

            outputs = model(batch)
            recon_loss = F.mse_loss(outputs.reconstructed, batch)
            rq_loss = outputs.rq_loss.mean()
            union_loss, union_pairs = supervised_contrastive_loss(
                outputs.z,
                batch_ids,
                adjacency,
                temperature=temperature,
                exclude_adjacency=exclude_adjacency,
            )
            total_loss = recon_loss + rq_loss + union_loss_weight * union_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            epoch_total += float(total_loss.item())
            epoch_recon += float(recon_loss.item())
            epoch_rq += float(rq_loss.item())
            epoch_union += float(union_loss.item())
            epoch_union_pairs += union_pairs

        n_batches = max(len(batches), 1)
        row = {
            "epoch": epoch + 1,
            "loss": epoch_total / n_batches,
            "recon": epoch_recon / n_batches,
            "rq": epoch_rq / n_batches,
            "union": epoch_union / n_batches,
            "union_pairs": epoch_union_pairs,
        }
        history.append(row)

        if epoch % 20 == 0 or epoch == epochs - 1:
            print(
                f"  task-aware rqvae epoch {epoch + 1}/{epochs} "
                f"loss={row['loss']:.4f} "
                f"recon={row['recon']:.4f} "
                f"rq={row['rq']:.4f} "
                f"union={row['union']:.4f} "
                f"pairs={row['union_pairs']}"
            )

    model = model.cpu()
    return model, {"history": history, "positive_pairs": len(positive_pairs)}


def quantize_task_aware(
    model: TaskAwareRQVAE,
    embeddings: np.ndarray,
    batch_size: int = 1024,
    device: str | None = None,
) -> np.ndarray:
    x = torch.tensor(embeddings, dtype=torch.float32)
    device = device or next(model.parameters()).device
    model = model.to(device)
    model.eval()
    all_indices = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            batch = x[start:start + batch_size].to(device)
            z = model.encode(batch)
            _, indices, _ = model.rqvae(z)
            all_indices.append(indices.cpu())
    return torch.cat(all_indices, dim=0).numpy()


def save_task_aware_rqvae_artifact(
    model: TaskAwareRQVAE,
    path: str,
    *,
    extra_metadata: dict | None = None,
) -> None:
    payload = {
        "artifact_version": TASK_AWARE_RQVAE_ARTIFACT_VERSION,
        "config": {
            "input_dim": model.input_dim,
            "latent_dim": model.latent_dim,
            "hidden_dims": model.hidden_dims,
            "num_quantizers": model.num_quantizers,
            "codebook_size": model.codebook_size,
            "commitment_weight": model.commitment_weight,
            "kmeans_iters": model.kmeans_iters,
            "dropout": model.dropout,
            "l2_normalize_latent": model.l2_normalize_latent,
        },
        "state_dict": model.cpu().state_dict(),
        "metadata": extra_metadata or {},
    }
    torch.save(payload, path)


def load_task_aware_rqvae_artifact(path: str, device: str | None = None) -> tuple[TaskAwareRQVAE, dict]:
    payload = torch.load(path, map_location=device or "cpu")
    cfg = payload["config"]
    model = TaskAwareRQVAE(
        input_dim=cfg["input_dim"],
        latent_dim=cfg["latent_dim"],
        hidden_dims=cfg["hidden_dims"],
        num_quantizers=cfg["num_quantizers"],
        codebook_size=cfg["codebook_size"],
        commitment_weight=cfg["commitment_weight"],
        dropout=cfg["dropout"],
        l2_normalize_latent=cfg.get("l2_normalize_latent", True),
        kmeans_iters=cfg.get("kmeans_iters") or 100,
    )
    model.load_state_dict(payload["state_dict"])
    model = model.to(device or "cpu")
    model.eval()
    return model, payload
