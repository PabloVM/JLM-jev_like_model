import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Qwen3Model


class BilinearDecisionHead(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.bilinear = nn.Bilinear(hidden_size, hidden_size, 1)

    def forward(self, answer_embedding, choice_embeddings):
        """
        answer_embedding: (batch_size, hidden_size)
        choice_embeddings: (batch_size, num_choices, hidden_size)
        """
        _, num_choices, _ = choice_embeddings.shape

        answer_expanded = answer_embedding.unsqueeze(1).expand(
            -1, num_choices, -1
        )  # (batch_size,hidden_size) -> (batch_size, num_choices, hidden_size)

        scores = self.bilinear(
            answer_expanded, choice_embeddings
        )  # Output: (batch_size, num_choices,1)
        scores = scores.squeeze(-1)  # (batch_size, num_choices)

        return scores


class JLM(nn.Module):
    def __init__(self, model_name="Qwen/Qwen3-0.6B") -> None:
        super().__init__()

        self.base_model = Qwen3Model.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
        )

        self.head = BilinearDecisionHead(self.base_model.config.hidden_size)

    def forward(self, input_ids, position_ids, attention_mask, choice_idx, answer_idx):
        with torch.no_grad():
            output = self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=None,
                input_embeds=None,
                use_cache=False,
            )

        hidden = output.last_hidden_state.float()
        self.head.to(hidden.device)

        batch_size = hidden.shape[0]

        logits_list = []

        for i in range(batch_size):
            c_indices = choice_idx[i]
            a_index = answer_idx[i]

            choice_embeds = hidden[i, c_indices].unsqueeze(
                0
            )  # (1, num_choices, hidden_size)
            answer_embed = hidden[i, a_index].unsqueeze(0)  # (1, hidden_size)

            scores = self.head(answer_embed, choice_embeds)  # (1, num_choices)
            logits_list.append(scores.squeeze(0))

        return logits_list

    @torch.no_grad
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
        attention_mask = (
            encoded["attention_mask"]
            .unsqueeze(0)
            .unsqueeze(0)
            .to(dtype=torch.float32)
            .to(device)
        )

        logits_list = self.forward(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            choice_idx=[encoded["choices_end_indxs"]],
            answer_idx=[encoded["answer_end_indx"]],
        )

        logits = logits_list[0]
        probs = F.softmax(logits, dim=-1)

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
