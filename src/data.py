import random

import torch
from torch.utils.data import Dataset, Sampler
from transformers import AutoTokenizer


class JLMTokenizer:
    def __init__(
        self, model_name="Qwen/Qwen3-0.6B", max_prompt_length=384, max_option_length=64
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_prompt_length = max_prompt_length
        self.max_option_length = max_option_length
        self.choice_begin_token = self.tokenizer.encode("<option>")
        self.choice_end_token = self.tokenizer.encode("</option>")
        self.answer_prompt_tokens = self.tokenizer.encode("Answer is:")
        self.newline_tokens = self.tokenizer.encode("\n")

    def _encode_prompt(self, state: str, question: str) -> list[int]:
        """Encodes `state\nquestion`, truncating the state (never the question) so
        the prompt fits in max_prompt_length."""
        question_tokens = self.tokenizer.encode(
            question, truncation=True, max_length=self.max_prompt_length
        )
        state_budget = (
            self.max_prompt_length - len(question_tokens) - len(self.newline_tokens)
        )
        if not state or state_budget <= 0:
            return question_tokens

        state_tokens = self.tokenizer.encode(
            state, truncation=True, max_length=state_budget
        )
        return state_tokens + self.newline_tokens + question_tokens

    def _create_input_tokens(self, state: str, question: str, choices: list[str]):
        """Converts the input and identifies the indices for each option and answer."""
        prompt_tokens = self._encode_prompt(state, question)
        choice_tokens = [
            self.tokenizer.encode(c, truncation=True, max_length=self.max_option_length)
            for c in choices
        ]

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

    def sequence_length(self, state: str, question: str, choices: list[str]) -> int:
        """Length of the encoded example, without building masks (for bucketing)."""
        input_ids, _ = self._create_input_tokens(state, question, choices)
        return len(input_ids)

    def encode_example(self, state: str, question: str, choices: list[str]):

        input_ids, idxs = self._create_input_tokens(state, question, choices)
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

    @staticmethod
    def _choices(item) -> list[str]:
        return ["no", "yes"] if item.get("kind", "choice") == "noul" else item["options"]

    def lengths(self) -> list[int]:
        """Encoded length of every example, used to group batches by length."""
        return [
            self.tokenizer.sequence_length(
                item.get("state") or "", item["question"], self._choices(item)
            )
            for item in self.dataset
        ]

    def __getitem__(self, idx):
        item = self.dataset[idx]
        state = item.get("state") or ""
        question = item["question"]
        kind = item.get("kind", "choice")
        choices = self._choices(item)

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
    max_choices = max(len(item["choices_end_indxs"]) for item in batch)

    padded_input_ids = torch.zeros((batch_size, max_len), dtype=torch.long)
    padded_position_ids = torch.zeros((batch_size, max_len), dtype=torch.long)

    # (batch_size, 1, max_len, max_len)
    padded_attn_mask = torch.full((batch_size, 1, max_len, max_len), -float("inf"))
    padded_choices_idx = torch.zeros((batch_size, max_choices), dtype=torch.long)
    choice_mask = torch.zeros((batch_size, max_choices), dtype=torch.bool)

    labels = []
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
        # Padding rows attend to themselves so no row is fully -inf (avoids NaNs)
        pad_idx = torch.arange(seq_len, max_len)
        padded_attn_mask[i, 0, pad_idx, pad_idx] = 0.0

        labels.append(item["label"])
        batch_answer_end_indxs.append(item["answer_end_indx"])

        num_choices = len(item["choices_end_indxs"])
        padded_choices_idx[i, :num_choices] = torch.tensor(item["choices_end_indxs"])
        choice_mask[i, :num_choices] = True

    return {
        "input_ids": padded_input_ids,
        "position_ids": padded_position_ids,
        "attention_mask": padded_attn_mask,
        "labels": torch.tensor(labels, dtype=torch.long),
        "choices_end_indxs": padded_choices_idx,
        "choice_mask": choice_mask,
        "answer_end_indxs": torch.tensor(batch_answer_end_indxs, dtype=torch.long),
    }


class LengthGroupedBatchSampler(Sampler[list[int]]):
    """Yields batches of examples with similar length to minimise padding.

    With shuffle=True, indices are shuffled, split into groups of
    batch_size * group_mult, sorted by length inside each group and the
    resulting batches are shuffled. With shuffle=False batches follow the
    global length order (longest first, so OOM shows up immediately).
    """

    def __init__(self, lengths, batch_size, shuffle=True, group_mult=50, seed=0):
        self.lengths = list(lengths)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.group_mult = group_mult
        self.seed = seed
        self.epoch = 0

    def __len__(self):
        return (len(self.lengths) + self.batch_size - 1) // self.batch_size

    def _split(self, indices):
        return [
            indices[i : i + self.batch_size]
            for i in range(0, len(indices), self.batch_size)
        ]

    def __iter__(self):
        indices = list(range(len(self.lengths)))

        if not self.shuffle:
            indices.sort(key=lambda i: self.lengths[i], reverse=True)
            yield from self._split(indices)
            return

        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1
        rng.shuffle(indices)

        group_size = self.batch_size * self.group_mult
        batches = []
        for g in range(0, len(indices), group_size):
            group = sorted(
                indices[g : g + group_size], key=lambda i: self.lengths[i], reverse=True
            )
            batches.extend(self._split(group))
        rng.shuffle(batches)
        yield from batches
