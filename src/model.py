import torch
import torch.nn as nn
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
        output = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=None,
            input_embeds=None,
            use_cache=False,
        )

        hidden = output.last_hidden_state.float()
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
