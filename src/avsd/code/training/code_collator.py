from typing import Any

import torch

from avsd.code.data.code_dataset import normalize_code_example
from avsd.code.prompts.student_prompt import build_student_prompt
from avsd.code.prompts.view_prompts import build_teacher_prompt, normalize_code_views


class CodeAVSDDataCollator:
    """Collate prepared code examples into student and static teacher prompts."""

    def __init__(
        self,
        tokenizer,
        *,
        max_length: int = 8192,
        views: str | tuple[str, ...] = ("reference", "hint", "feedback"),
        student_enable_thinking: bool = False,
        teacher_enable_thinking: bool = True,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.views = normalize_code_views(views)
        self.student_enable_thinking = student_enable_thinking
        self.teacher_enable_thinking = teacher_enable_thinking
        self.tokenizer.padding_side = "right"

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        examples = [normalize_code_example(feature) for feature in features]
        student_prompts = [
            build_student_prompt(
                self.tokenizer,
                example,
                enable_thinking=self.student_enable_thinking,
            )
            for example in examples
        ]
        student_encoded, student_lengths = self._tokenize(student_prompts)
        batch: dict[str, Any] = {
            "student_prompts": student_encoded["input_ids"],
            "student_prompt_attention_mask": student_encoded["attention_mask"],
            "student_prompt_length": max(student_lengths),
            "student_prompt_lengths_per_example": torch.tensor(student_lengths, dtype=torch.long),
            "raw_examples": [example.to_dict() for example in examples],
        }

        for view in self.views:
            if view == "feedback":
                continue
            teacher_prompts = [
                build_teacher_prompt(
                    self.tokenizer,
                    example,
                    view,
                    enable_thinking=self.teacher_enable_thinking,
                )
                for example in examples
            ]
            encoded, lengths = self._tokenize(teacher_prompts)
            batch[f"teacher_{view}_prompts"] = encoded["input_ids"]
            batch[f"teacher_{view}_prompt_attention_mask"] = encoded["attention_mask"]
            batch[f"teacher_{view}_prompt_length"] = max(lengths)
            batch[f"teacher_{view}_prompt_lengths_per_example"] = torch.tensor(lengths, dtype=torch.long)
        return batch

    def _tokenize(self, prompts: list[str]) -> tuple[dict[str, torch.Tensor], list[int]]:
        encoded_no_pad = self.tokenizer(
            prompts,
            padding=False,
            truncation=True,
            max_length=self.max_length,
        )
        lengths = [len(ids) for ids in encoded_no_pad["input_ids"]]
        max_len = max(lengths) if lengths else 0
        encoded = self.tokenizer(
            prompts,
            padding="max_length",
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        )
        return encoded, lengths
