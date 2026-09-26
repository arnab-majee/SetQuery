from pathlib import Path
from threading import Lock
import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from qwen_vl_utils import process_vision_info

from core.config import settings
from core.exception import ModelServiceError

class QwenVLService:
    def __init__(self) -> None:
        self.model = None
        self.processor = None
        self._lock = Lock()

    @property
    def loaded(self) -> bool:
        return self.model is not None and self.processor is not None

    def load(self) -> None:
        if self.loaded:
            return
        with self._lock:
            if self.loaded:
                return
            try:
                self.processor = AutoProcessor.from_pretrained(settings.model_name)
                self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    settings.model_name, torch_dtype="auto", device_map="auto"
                )
                self.model.eval()
            except Exception as exc:
                self.model = None
                self.processor = None
                raise ModelServiceError(
                    "Qwen2.5-VL could not be loaded. Check the model name, dependencies, RAM/VRAM and PyTorch installation. "
                    f"Original error: {exc}"
                ) from exc

    def generate(
            self, 
            image_path: Path, 
            question: str, 
            max_new_tokens: int | None = None) -> str:

        if not image_path.exists():
            raise FileNotFoundError(f"Image file does not exist: {image_path}")
        self.load()
        token_limit = max_new_tokens if max_new_tokens is not None else settings.max_new_tokens
        messages = [{"role": "user", "content": [{"type": "image", "image": str(image_path)}, {"type": "text", "text": question}]}]
        try:
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
            input_device = self.model.get_input_embeddings().weight.device
            inputs = {key: value.to(input_device) if hasattr(value, "to") else value for key, value in inputs.items()}
            with torch.inference_mode():
                generated_ids = self.model.generate(**inputs, max_new_tokens=token_limit, do_sample=False)
            input_ids = inputs["input_ids"]
            generated_ids_trimmed = [output_ids[len(input_ids[index]):] for index, output_ids in enumerate(generated_ids)]
            return self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()
        except Exception as exc:
            raise ModelServiceError(f"Qwen2.5-VL inference failed: {exc}") from exc

qwen_vl_service = QwenVLService()
