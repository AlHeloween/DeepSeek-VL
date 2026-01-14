# Copyright (c) 2023-2024 DeepSeek.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

from deepseek_vl.models import MultiModalityCausalLM, VLChatProcessor
from deepseek_vl.utils.io import load_pil_images

# specify the path to the model
model_path = "deepseek-ai/deepseek-vl-1.3b-base"
vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
tokenizer = vl_chat_processor.tokenizer

_CUDA_AVAILABLE = torch.cuda.is_available()
_VRAM_BYTES = torch.cuda.get_device_properties(0).total_memory if _CUDA_AVAILABLE else 0
_SMALL_VRAM = _CUDA_AVAILABLE and _VRAM_BYTES < (8 * 1024 * 1024 * 1024)

# On small GPUs (e.g. 4GB), `device_map="auto"` will often offload to CPU RAM.
# This is slow but keeps the script usable without hard-forcing CPU-only mode.
_DEVICE_MAP = "auto" if _CUDA_AVAILABLE else None

# Pascal (and older) do not support bf16; prefer fp16 on CUDA, fp32 on CPU.
_DTYPE = torch.float16 if _CUDA_AVAILABLE else torch.float32
vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    device_map=_DEVICE_MAP,
    torch_dtype=_DTYPE,
)
vl_gpt = vl_gpt.eval()

print(
    "DeepSeek-VL load:",
    f"cuda_available={_CUDA_AVAILABLE}",
    f"vram_gb={(_VRAM_BYTES / (1024**3)):.2f}" if _CUDA_AVAILABLE else "vram_gb=n/a",
    f"device_map={_DEVICE_MAP!r}",
    f"dtype={_DTYPE}",
)
if hasattr(vl_gpt, "hf_device_map"):
    print("hf_device_map:", getattr(vl_gpt, "hf_device_map"))

# single image conversation example
_HERE = Path(__file__).resolve().parent
conversation = [
    {
        "role": "User",
        "content": "<image_placeholder>Describe each stage of this image.",     
        "images": [str(_HERE / "images" / "training_pipelines.jpg")],
    },
    {"role": "Assistant", "content": ""},
]

# multiple images (or in-context learning) conversation example
# conversation = [
#     {
#         "role": "User",
#         "content": "<image_placeholder>A dog wearing nothing in the foreground, "
#                    "<image_placeholder>a dog wearing a santa hat, "
#                    "<image_placeholder>a dog wearing a wizard outfit, and "
#                    "<image_placeholder>what's the dog wearing?",
#         "images": [
#             str(_HERE / "images" / "dog_a.png"),
#             str(_HERE / "images" / "dog_b.png"),
#             str(_HERE / "images" / "dog_c.png"),
#             str(_HERE / "images" / "dog_d.png"),
#         ],
#     },
#     {"role": "Assistant", "content": ""}
# ]

# load images and prepare for inputs
pil_images = load_pil_images(conversation)
prepare_inputs = vl_chat_processor(
    conversations=conversation, images=pil_images, force_batchify=True
).to(vl_gpt.device, dtype=_DTYPE if _CUDA_AVAILABLE else torch.float32)

# run image encoder to get the image embeddings
inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)

# run the model to get the response
outputs = vl_gpt.language_model.generate(
    inputs_embeds=inputs_embeds,
    attention_mask=prepare_inputs.attention_mask,
    pad_token_id=tokenizer.eos_token_id,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
    max_new_tokens=512,
    do_sample=False,
    use_cache=True,
)

answer = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
print(f"{prepare_inputs['sft_format'][0]}", answer)
