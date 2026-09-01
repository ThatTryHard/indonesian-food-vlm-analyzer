"""Pinned Qwen2-VL inference wrapper used by the sequential Kaggle notebook."""

from __future__ import annotations

import gc
from pathlib import Path

from PIL import Image

from .vlm import build_visible_prompt, parse_vlm_output


class QwenVisibleIngredientAnalyzer:
    def __init__(self, ontology, model_id: str, revision: str, max_image_side: int = 512):
        import torch
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

        if not torch.cuda.is_available():
            raise RuntimeError("Qwen2-VL evaluation requires a CUDA GPU in this portfolio pipeline")
        self.torch = torch
        self.ontology = ontology
        self.max_image_side = max_image_side
        self.prompt = build_visible_prompt(ontology)
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_id,
            revision=revision,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(model_id, revision=revision, use_fast=False)

    @property
    def device(self):
        return next(self.model.parameters()).device

    def infer(self, image_path: str | Path, max_new_tokens: int = 256) -> dict[str, object]:
        from qwen_vl_utils import process_vision_info

        with Image.open(image_path) as raw:
            image = raw.convert("RGB")
            image.thumbnail((self.max_image_side, self.max_image_side))
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image.copy()},
                        {"type": "text", "text": self.prompt},
                    ],
                }
            ]
        chat_text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[chat_text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.device)
        with self.torch.inference_mode():
            generated = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True)
        trimmed = [output[len(input_ids) :] for input_ids, output in zip(inputs.input_ids, generated, strict=True)]
        raw_output = self.processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[
            0
        ]
        parsed = parse_vlm_output(raw_output, self.ontology)
        del inputs, generated, trimmed, image_inputs, video_inputs
        gc.collect()
        self.torch.cuda.empty_cache()
        return {
            "raw_output": raw_output,
            "food_name": parsed.food_name,
            "visible_ingredients": self.ontology.serialize(parsed.visible_ingredients),
            "uncertain_ingredients": self.ontology.serialize(parsed.uncertain_ingredients),
            "unknown_labels": "|".join(parsed.unknown_labels),
            "abstain": parsed.abstain,
            "reason": parsed.reason,
            "parse_ok": parsed.parse_ok,
            "parse_error": parsed.parse_error,
        }
