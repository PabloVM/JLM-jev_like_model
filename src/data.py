import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class JLMTokenizer:
    def __init__(self, model_name="Qwen/Qwen3-0.6B"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.choice_begin_token = self.tokenizer.encode("<option>")
        self.choice_end_token = self.tokenizer.encode("</option>")
        self.answer_prompt_tokens = self.tokenizer.encode("Answer is:")

    def _create_input_tokens(self, prompt: str, choices: list[str]):
        """Converts the input and identifies the indices for each option and answer."""
        prompt_tokens = self.tokenizer.encode(prompt)
        choice_tokens = [self.tokenizer.encode(c) for c in choices]

        final_tokens = list(prompt_tokens)
        idxs = {"question_end_indx": len(final_tokens) - 1, "choices_end_indxs": []}

        for choice in choice_tokens:
            final_tokens.extend(self.choice_begin_token)
            final_tokens.extend(choice)
            final_tokens.extend(self.choice_end_token)
            idxs["choices_end_indxs"].append(len(final_tokens) - 1)

        final_tokens.extend(self.answer_prompt_tokens)
        idxs["answer_end_indx"] = len(final_tokens) - 1

        return final_tokens, idxs

    def _generate_position_ids(self, idxs: dict) -> list[int]:
        """Generates position_ids restarting position for each choice."""
        position_ids = [idx for idx in range(idxs["question_end_indx"] + 1)]

        start_idx = len(position_ids)
        choice_start_idx = start_idx

        for choice_end_idx in idxs["choices_end_indxs"]:
            choice_len = (choice_end_idx - choice_start_idx) + 1
            position_ids.extend(range(start_idx, start_idx + choice_len))
            choice_start_idx = choice_end_idx + 1

        answer_len = (idxs["answer_end_indx"] + 1) - (idxs["choices_end_indxs"][-1] + 1)
        position_ids.extend(range(start_idx, start_idx + answer_len))

        return position_ids

    def _get_attention_mask(self, idxs: dict) -> torch.Tensor:
        """Generates a 2D Mask (dim, dim) preventing choices from seeing each other."""
        dim = idxs["answer_end_indx"] + 1
        mask = torch.zeros((dim, dim))
        q_len = idxs["question_end_indx"] + 1

        # Question causal attention
        mask[:q_len, :q_len] = torch.tril(torch.ones(q_len, q_len))

        # Each choice can see question and itself
        choice_start = q_len
        for choice_end in idxs["choices_end_indxs"]:
            c_len = choice_end - choice_start + 1
            mask[choice_start : choice_end + 1, :q_len] = 1
            mask[choice_start : choice_end + 1, choice_start : choice_end + 1] = (
                torch.tril(torch.ones(c_len, c_len))
            )
            choice_start = choice_end + 1

        # Answer can see all
        ans_start = idxs["choices_end_indxs"][-1] + 1
        mask[ans_start:, :ans_start] = 1
        mask[ans_start:, ans_start:] = torch.tril(
            torch.ones(dim - ans_start, dim - ans_start)
        )

        # Convert 1 -> 0.0 and 0 -> -inf
        mask = torch.where(mask == 1.0, 0.0, -float("inf"))
        return mask  #  (dim, dim)

    def encode_example(self, state: str, question: str, choices: list[str]):

        prompt_text = f"{state}\n{question}" if state else question
        input_ids, idxs = self._create_input_tokens(prompt_text, choices)
        position_ids = self._generate_position_ids(idxs)
        attention_mask = self._get_attention_mask(idxs)

        return {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "choices_end_indxs": idxs["choices_end_indxs"],
            "answer_end_indx": idxs["answer_end_indx"],
        }


class JLMDataset(Dataset):
    def __init__(self, hf_dataset, tokenizer):
        self.dataset = hf_dataset
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        state = item.get("state", "")
        question = item["question"]
        choices = item["options"]
        kind = item.get("kind", "choice")

        if kind == "noul":
            choices = ["no", "yes"]
        else:
            choices = item["options"]

        target_list = item["target"]

        # One-Hot Encoding

        if isinstance(target_list, list):
            correct_index = target_list.index(max(target_list))
        else:
            correct_index = int(target_list)

        encoded = self.tokenizer.encode_example(state, question, choices)
        encoded["label"] = correct_index
        encoded["kind"] = kind

        return encoded


def jlm_collate_fn(batch):
    """Creates a batch applying padding."""
    batch_size = len(batch)

    max_len = max(len(item["input_ids"]) for item in batch)

    padded_input_ids = torch.zeros((batch_size, max_len), dtype=torch.long)
    padded_position_ids = torch.zeros((batch_size, max_len), dtype=torch.long)

    # (batch_size, 1, max_len, max_len)
    padded_attn_mask = torch.full((batch_size, 1, max_len, max_len), -float("inf"))

    labels = []
    batch_choices_end_indxs = []
    batch_answer_end_indxs = []

    for i, item in enumerate(batch):
        seq_len = len(item["input_ids"])

        padded_input_ids[i, :seq_len] = torch.tensor(
            item["input_ids"], dtype=torch.long
        )
        padded_position_ids[i, :seq_len] = torch.tensor(
            item["position_ids"], dtype=torch.long
        )

        padded_attn_mask[i, 0, :seq_len, :seq_len] = item["attention_mask"]

        labels.append(item["label"])
        batch_choices_end_indxs.append(item["choices_end_indxs"])
        batch_answer_end_indxs.append(item["answer_end_indx"])

    return {
        "input_ids": padded_input_ids,
        "position_ids": padded_position_ids,
        "attention_mask": padded_attn_mask,
        "labels": torch.tensor(labels, dtype=torch.long),
        "choices_end_indxs": batch_choices_end_indxs,
        "answer_end_indxs": batch_answer_end_indxs,
    }
