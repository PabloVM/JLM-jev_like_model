import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from transformers import Qwen3Model


class DecisionHead(nn.Module):
    def __init__(self, hidden_size, proj_dim=256):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.proj_answer = nn.Linear(hidden_size, proj_dim)
        self.proj_choices = nn.Linear(hidden_size, proj_dim)
        self.logit_scale = nn.Parameter(
            torch.tensor(math.log(10.0))
        )  # Learnable temperature

    def forward(self, answer_embedding, choice_embeddings):
        """
        answer_embedding: (batch_size, hidden_size)
        choice_embeddings: (batch_size, num_choices, hidden_size)
        """

        a = F.normalize(self.proj_answer(self.norm(answer_embedding)), dim=1)
        c = F.normalize(self.proj_choices(self.norm(choice_embeddings)))

        cos = torch.bmm(c, a.unsqueeze(-1)).squeeze(-1)  # (B,N)
        scale = self.logit_scale.exp().clamp(max=100)

        return scale * cos


def default_backbone_dtype() -> torch.dtype:
    """bf16 only on GPUs with native support (Ampere+). T4/P100 report bf16 as
    supported through emulation, which is several times slower than fp16."""
    if not torch.cuda.is_available():
        return torch.float32
    if torch.cuda.get_device_capability()[0] >= 8:
        return torch.bfloat16
    return torch.float16


class JLM(nn.Module):
    def __init__(self, model_name="Qwen/Qwen3-0.6B", dtype=None) -> None:
        super().__init__()

        self.backbone_dtype = dtype or default_backbone_dtype()

        self.base_model = Qwen3Model.from_pretrained(
            model_name,
            torch_dtype=self.backbone_dtype,
        )
        self.base_model.requires_grad_(False)

        self.head = DecisionHead(self.base_model.config.hidden_size)

    @torch.no_grad()
    def encode(self, input_ids, position_ids, attention_mask, choice_idx, answer_idx):
        """Runs the frozen backbone and returns the fp32 embeddings the head needs.

        Returns answer_embed (batch_size, hidden_size) and
        choice_embeds (batch_size, num_choices, hidden_size).
        """
        output = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask.to(self.backbone_dtype),
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=None,
            use_cache=False,
        )

        hidden = output.last_hidden_state.float()
        batch_size = hidden.shape[0]

        batch_indxs = torch.arange(batch_size, device=hidden.device)
        answer_embed = hidden[batch_indxs, answer_idx]
        choice_embeds = hidden[batch_indxs.unsqueeze(1), choice_idx]
        return answer_embed, choice_embeds

    def score(self, answer_embed, choice_embeds, choice_mask):
        """Applies the decision head in fp32, even inside an autocast region."""
        with torch.autocast(device_type=answer_embed.device.type, enabled=False):
            scores = self.head(answer_embed.float(), choice_embeds.float())
        return scores.masked_fill(~choice_mask, -float("inf"))

    def forward(
        self,
        input_ids,
        position_ids,
        attention_mask,
        choice_idx,
        choice_mask,
        answer_idx,
    ):
        answer_embed, choice_embeds = self.encode(
            input_ids, position_ids, attention_mask, choice_idx, answer_idx
        )
        return self.score(answer_embed, choice_embeds, choice_mask)

    @torch.no_grad()
    def predict(
        self,
        tokenizer,
        question: str,
        choices: list[str] | None = None,
        state: str = "",
        kind: str = "choice",
    ) -> dict:
        """
        High-level method for inference the 3 main primitives of JEV models
        """
        self.eval()
        device = next(self.parameters()).device

        if kind == "noul":
            choices = ["no", "yes"]
        elif choices is None or len(choices) == 0:
            raise ValueError(
                f"For kind {kind} it is necessary to provide a list of choices."
            )

        encoded = tokenizer.encode_example(state, question, choices)

        input_ids = (
            torch.tensor(encoded["input_ids"], dtype=torch.long).unsqueeze(0).to(device)
        )
        position_ids = (
            torch.tensor(encoded["position_ids"], dtype=torch.long)
            .unsqueeze(0)
            .to(device)
        )
        attention_mask = encoded["attention_mask"].unsqueeze(0).unsqueeze(0).to(device)

        num_choices = len(encoded["choices_end_indxs"])

        choice_idx = (
            torch.tensor(encoded["choices_end_indxs"], dtype=torch.long)
            .unsqueeze(0)
            .to(device)
        )
        choice_mask = torch.ones((1, num_choices), dtype=torch.bool).to(device)
        answer_idx = torch.tensor([encoded["answer_end_indx"]], dtype=torch.long).to(
            device
        )

        logits = self.forward(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            choice_idx=choice_idx,
            choice_mask=choice_mask,
            answer_idx=answer_idx,
        )

        probs = F.softmax(logits[0], dim=-1)

        if kind == "noul":
            p_yes = probs[1].item() if len(probs) > 1 else probs[0].item()
            return {
                "kind": "noul",
                "noul": round(p_yes, 4),
            }

        elif kind == "score":
            levels = torch.arange(
                1, len(choices) + 1, device=device, dtype=torch.float32
            )
            expected_score = torch.sum(levels * probs).item()
            return {
                "kind": "score",
                "score": round(expected_score, 2),
                "probabilities": {
                    choice: round(p.item(), 4) for choice, p in zip(choices, probs)
                },
            }

        else:
            best_idx = int(torch.argmax(probs).item())
            return {
                "kind": "choice",
                "choice": choices[best_idx],
                "probabilities": {
                    choice: round(p.item(), 4) for choice, p in zip(choices, probs)
                },
            }
