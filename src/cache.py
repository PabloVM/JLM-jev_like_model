"""Precomputes the frozen backbone embeddings once so the decision head can be
trained for many epochs without running Qwen again."""

from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.data import JLMDataset, LengthGroupedBatchSampler, jlm_collate_fn


@torch.no_grad()
def build_embedding_cache(
    model,
    dataset: JLMDataset,
    batch_size: int = 32,
    num_workers: int = 2,
    device: str | torch.device = "cuda",
    path: str | Path | None = None,
    store_dtype: torch.dtype = torch.float16,
) -> dict:
    """Runs the backbone over `dataset` and returns (and optionally saves) a dict:

    answer_embeds (N, hidden), choice_embeds (N, max_choices, hidden),
    choice_mask (N, max_choices) and labels (N), in dataset order.
    If `path` exists, the cache is loaded from disk instead.
    """
    if path is not None and Path(path).exists():
        print(f"Loading embedding cache from {path}")
        return torch.load(path)

    model.eval()
    sampler = LengthGroupedBatchSampler(dataset.lengths(), batch_size, shuffle=False)
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=jlm_collate_fn,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    n = len(dataset)
    hidden_size = model.base_model.config.hidden_size
    answer_embeds = torch.empty((n, hidden_size), dtype=store_dtype)
    labels = torch.empty(n, dtype=torch.long)
    choice_chunks = []  # (indices, choice_embeds, choice_mask) per batch
    non_finite = 0

    for batch_indices, batch in zip(
        tqdm(list(sampler), desc="Caching embeddings"), loader
    ):
        answer_embed, choice_embeds = model.encode(
            input_ids=batch["input_ids"].to(device, non_blocking=True),
            position_ids=batch["position_ids"].to(device, non_blocking=True),
            attention_mask=batch["attention_mask"].to(device, non_blocking=True),
            choice_idx=batch["choices_end_indxs"].to(device, non_blocking=True),
            answer_idx=batch["answer_end_indxs"].to(device, non_blocking=True),
        )
        answer_embed = answer_embed.to(store_dtype).cpu()
        choice_embeds = choice_embeds.to(store_dtype).cpu()

        finite = torch.isfinite(answer_embed).all(-1) & torch.isfinite(
            choice_embeds
        ).flatten(1).all(-1)
        non_finite += int((~finite).sum())

        idx = torch.tensor(batch_indices)
        answer_embeds[idx] = answer_embed
        labels[idx] = batch["labels"]
        choice_chunks.append((idx, choice_embeds, batch["choice_mask"]))

    max_choices = max(c.shape[1] for _, c, _ in choice_chunks)
    choice_embeds_all = torch.zeros((n, max_choices, hidden_size), dtype=store_dtype)
    choice_mask_all = torch.zeros((n, max_choices), dtype=torch.bool)
    for idx, choices, mask in choice_chunks:
        choice_embeds_all[idx, : choices.shape[1]] = choices
        choice_mask_all[idx, : mask.shape[1]] = mask

    if non_finite:
        print(
            f"WARNING: {non_finite}/{n} examples have NaN/inf embeddings. "
            "If the backbone runs in fp16, try JLM(..., dtype=torch.float32)."
        )

    cache = {
        "answer_embeds": answer_embeds,
        "choice_embeds": choice_embeds_all,
        "choice_mask": choice_mask_all,
        "labels": labels,
    }
    if path is not None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(cache, path)
        print(f"Embedding cache saved to {path}")
    return cache


def iterate_cached_batches(cache: dict, batch_size: int, shuffle: bool = True):
    """Yields (answer_embeds, choice_embeds, choice_mask, labels) batches from a
    cache, on whatever device the cache tensors live (move it to GPU once)."""
    n = cache["labels"].shape[0]
    device = cache["labels"].device
    order = torch.randperm(n, device=device) if shuffle else torch.arange(n, device=device)

    for start in range(0, n, batch_size):
        idx = order[start : start + batch_size]
        choice_mask = cache["choice_mask"][idx]
        # Trim choice slots that no example in this batch uses
        num_choices = int(choice_mask.sum(-1).max())
        yield (
            cache["answer_embeds"][idx],
            cache["choice_embeds"][idx, :num_choices],
            choice_mask[:, :num_choices],
            cache["labels"][idx],
        )


def cache_to(cache: dict, device) -> dict:
    return {k: v.to(device) for k, v in cache.items()}
