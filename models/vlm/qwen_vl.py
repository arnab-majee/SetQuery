
import torch
from pathlib import Path
from typing import Any

from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from core.config import settings
from core.exception import ModelServiceError

try:
    from qwen_vl_utils import process_vision_info
except ImportError:
    process_vision_info = None


class QwenVLService:
    def __init__(self):
        self.model = None
        self.processor = None
        self.loaded = False

    def load(self) -> None:
        if self.loaded:
            return

        if process_vision_info is None:
            raise ModelServiceError(
                "qwen-vl-utils is not installed."
            )

        try:
            self.processor = AutoProcessor.from_pretrained(settings.model_name)

            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                settings.model_name,
                torch_dtype="auto",
                device_map="auto",
            )

            self.model.eval()
            self.loaded = True

        except Exception as exc:
            raise ModelServiceError(f"Failed to load Qwen2.5-VL: {exc}") from exc

    def generate( self, image_path: Path, question: str, max_new_tokens: int | None = None) -> str:

        if not self.loaded:
            self.load()

        if self.model is None or self.processor is None:
            raise ModelServiceError("Qwen2.5-VL service is not initialized.")

        if not image_path.exists():
            raise ValueError( f"Image does not exist: {image_path}")

        token_limit = (
            max_new_tokens
            if max_new_tokens is not None
            else settings.max_new_tokens
        )

        messages: list[dict[str, Any]] = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": str(image_path),
                    },
                    {
                        "type": "text",
                        "text": question,
                    },
                ],
            }
        ]

        try:
            text = self.processor.apply_chat_template(messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            image_inputs, video_inputs = process_vision_info(messages)

            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )

            input_device = (
                self.model.get_input_embeddings()
                .weight.device
            )

            inputs = {
                key: value.to(input_device)
                if hasattr(value, "to")
                else value
                for key, value in inputs.items()
            }

            with torch.inference_mode():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=token_limit,
                    do_sample=False,
                )

            generated_ids_trimmed = [
                out_ids[len(in_ids):]
                for in_ids, out_ids in zip(
                    inputs["input_ids"],
                    generated_ids,
                )
            ]

            output_text = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )

            if not output_text:
                return ""

            return output_text[0].strip()

        except Exception as exc:
            raise ModelServiceError(
                f"Qwen2.5-VL inference failed: {exc}"
            ) from exc


qwen_vl_service = QwenVLService()
            
