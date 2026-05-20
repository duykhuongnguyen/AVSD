import torch

from avsd.common.chat_template import apply_chat_template_compat
from avsd.math.privileged_views import build_teacher_user_message, get_view_payload


class SelfDistillationDataCollator:
    """
    Data collator for self-distillation that creates both student and teacher inputs.

    Student: sees only the problem (with chat template)
    Teacher: sees problem + solution + transition prompt (with chat template)

    To enable batch-level operations (like original GKD), we pad prompts to the same length
    within each batch, and track the actual (unpadded) prompt lengths for loss masking.
    """

    def __init__(
        self,
        tokenizer,
        max_length=2048,
        reason_first=True,
        multi_view_mode="single",
        single_view_pi="full_solution",
        pi_views=("full_solution", "partial_solution", "answer_only"),
        partial_solution_ratio=0.5,
        student_enable_thinking=False,
        teacher_enable_thinking=True,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.reason_first = reason_first
        self.multi_view_mode = multi_view_mode
        self.single_view_pi = single_view_pi
        self.pi_views = tuple(pi_views)
        self.partial_solution_ratio = partial_solution_ratio
        self.student_enable_thinking = student_enable_thinking
        self.teacher_enable_thinking = teacher_enable_thinking

        # Prompt for reasoning about the solution before teaching
        self.reason_first_prompt = (
            "\n\nThe reference reasoning above arrives at the correct answer. "
            "Please analyze this solution and explain the key reasoning steps and problem-solving strategies employed. "
            "Do NOT use <think> tags. Do NOT derive your own solution. "
            "Simply analyze and explain the reference solution provided above.\n"
        )
        # Prompt for transitioning to teaching mode after reasoning
        self.transition_prompt = (
            "\n\nAfter reading the reference solution above, make sure you truly understand "
            "the reasoning behind each step — do not copy or paraphrase it. Now, using your "
            "own words and independent reasoning, derive the same final answer to the problem above. "
            "Think step by step, explore different approaches, and don't be afraid to backtrack "
            "or reconsider if something doesn't work out:\n"
        )

        # Set padding side explicitly for consistency
        print(f"[DataCollator] Original padding_side: {self.tokenizer.padding_side}")
        self.tokenizer.padding_side = "right"
        print(f"[DataCollator] Set padding_side to: {self.tokenizer.padding_side}")
        print(f"[DataCollator] Reason first mode: {self.reason_first}")
        print(f"[DataCollator] Multi-view mode: {self.multi_view_mode}")
        print(f"[DataCollator] Single-view PI: {self.single_view_pi}")
        print(f"[DataCollator] Privileged views: {self.pi_views}")
        print(f"[DataCollator] Student thinking mode: {self.student_enable_thinking}")
        print(f"[DataCollator] Teacher thinking mode: {self.teacher_enable_thinking}")

    def _tokenize_prompt_batch(self, prompts):
        encoded_no_pad = self.tokenizer(
            prompts,
            padding=False,
            truncation=True,
            max_length=self.max_length,
        )
        prompt_lengths = [len(ids) for ids in encoded_no_pad["input_ids"]]
        max_prompt_len = max(prompt_lengths)
        encoded = self.tokenizer(
            prompts,
            padding="max_length",
            truncation=True,
            max_length=max_prompt_len,
            return_tensors="pt",
        )
        return encoded, prompt_lengths, max_prompt_len

    def __call__(self, features):

        batch_size = len(features)

        # Prepare student and teacher prompts using chat template (matching evaluation)
        student_prompts = []
        teacher_prompts = []
        teacher_prompts_by_view = {view: [] for view in self.pi_views}
        teacher_reasoning_prompts = []  # NEW: for reason_first mode

        for feature in features:
            # Extract problem and solution from dataset
            # Handle different possible column names
            problem = feature["problem"]
            solution = feature["solution"]

            # Student prompt: just the problem with instruction (matching evaluation format)
            student_user_message = f"Problem: {problem}\n\nPlease reason step by step, and put your final answer within \\boxed{{}}."
            student_messages = [{"role": "user", "content": student_user_message}]

            # Apply chat template for student (matching evaluation)
            student_prompt = apply_chat_template_compat(
                self.tokenizer,
                student_messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=self.student_enable_thinking,
            )
            student_prompts.append(student_prompt)

            if self.multi_view_mode != "single":
                for view in self.pi_views:
                    payload = get_view_payload(
                        problem=problem,
                        solution=solution,
                        view_type=view,
                        partial_solution_ratio=self.partial_solution_ratio,
                    )
                    teacher_user_message = build_teacher_user_message(
                        problem=problem,
                        payload=payload,
                        view_type=view,
                    )
                    teacher_messages = [{"role": "user", "content": teacher_user_message}]
                    teacher_prompt = apply_chat_template_compat(
                        self.tokenizer,
                        teacher_messages,
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=self.teacher_enable_thinking,
                    )
                    teacher_prompts_by_view[view].append(teacher_prompt)
            elif self.reason_first:
                # Reasoning prompt: ask teacher to analyze the solution
                reasoning_user_message = (
                    f"Problem: {problem}\n\n"
                    f"Here is a correct reasoning to this problem:"
                    f"=== Reference Reasoning Start ===\n"
                    f"{solution}\n"
                    f"=== Reference Reasoning End ===\n\n"
                    f"{self.reason_first_prompt}"
                )
                reasoning_messages = [{"role": "user", "content": reasoning_user_message}]
                reasoning_prompt = apply_chat_template_compat(
                    self.tokenizer,
                    reasoning_messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=self.teacher_enable_thinking,
                )
                teacher_reasoning_prompts.append(reasoning_prompt)

                # Teacher prompt will be constructed during training after reasoning
                # For now, create placeholder (will be replaced in training_step)
                teacher_prompts.append("")  # Placeholder
            else:
                if self.single_view_pi == "full_solution":
                    teacher_user_message = (
                        f"Problem: {problem}\n\n"
                        f"Here is a reference solution to this problem:\n"
                        f"=== Reference Solution Begin ===\n{solution}\n=== Reference Solution End ===\n"
                        f"{self.transition_prompt}\n"
                        f"Please reason step by step, and put your final answer within \\boxed{{}}."
                    )
                else:
                    payload = get_view_payload(
                        problem=problem,
                        solution=solution,
                        view_type=self.single_view_pi,
                        partial_solution_ratio=self.partial_solution_ratio,
                    )
                    teacher_user_message = (
                        f"Problem: {problem}\n\n"
                        "Here is reference information for this problem:\n"
                        f"=== Reference Information ({self.single_view_pi}) Begin ===\n{payload}\n"
                        f"=== Reference Information ({self.single_view_pi}) End ===\n"
                        f"{self.transition_prompt}\n"
                        f"Please reason step by step, and put your final answer within \\boxed{{}}."
                    )
                teacher_messages = [{"role": "user", "content": teacher_user_message}]

                # Apply chat template for teacher
                teacher_prompt = apply_chat_template_compat(
                    self.tokenizer,
                    teacher_messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=self.teacher_enable_thinking,
                )
                teacher_prompts.append(teacher_prompt)

        student_encoded, student_prompt_lengths, max_student_prompt_len = self._tokenize_prompt_batch(
            student_prompts
        )

        result = {
            "student_prompts": student_encoded["input_ids"],
            "student_prompt_attention_mask": student_encoded["attention_mask"],
            "student_prompt_length": max_student_prompt_len,  # Single value for batch!
            # Keep individual lengths for proper masking
            "student_prompt_lengths_per_example": torch.tensor(student_prompt_lengths),
        }

        if self.multi_view_mode != "single":
            for view, prompts in teacher_prompts_by_view.items():
                teacher_encoded, _, max_teacher_prompt_len = self._tokenize_prompt_batch(prompts)
                result.update(
                    {
                        f"teacher_{view}_prompts": teacher_encoded["input_ids"],
                        f"teacher_{view}_prompt_attention_mask": teacher_encoded["attention_mask"],
                        f"teacher_{view}_prompt_length": max_teacher_prompt_len,
                    }
                )
        elif self.reason_first:
            # Tokenize reasoning prompts
            reasoning_encoded, _, max_reasoning_prompt_len = self._tokenize_prompt_batch(
                teacher_reasoning_prompts
            )

            # Tokenize transition prompt (this will be appended after reasoning)
            # Don't use chat template here - just the raw text
            transition_text = f"\n{self.transition_prompt}\nPlease reason step by step, and put your final answer within \\boxed{{}}."
            transition_encoded = self.tokenizer(
                [transition_text] * batch_size,
                padding=False,
                truncation=False,
                return_tensors="pt",
            )

            result.update(
                {
                    "teacher_reasoning_prompts": reasoning_encoded["input_ids"],
                    "teacher_reasoning_attention_mask": reasoning_encoded["attention_mask"],
                    "teacher_reasoning_prompt_length": max_reasoning_prompt_len,
                    "teacher_transition_tokens": transition_encoded["input_ids"],
                }
            )
        else:
            # Normal mode: tokenize teacher prompts
            teacher_encoded, teacher_prompt_lengths, max_teacher_prompt_len = self._tokenize_prompt_batch(
                teacher_prompts
            )

            result.update(
                {
                    "teacher_prompts": teacher_encoded["input_ids"],
                    "teacher_prompt_attention_mask": teacher_encoded["attention_mask"],
                    "teacher_prompt_length": max_teacher_prompt_len,
                    "teacher_prompt_lengths_per_example": torch.tensor(teacher_prompt_lengths),
                }
            )

        return result
