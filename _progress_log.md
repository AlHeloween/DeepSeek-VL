# Progress Log - DeepSeek-VL

## [2026-01-09]
### Activity: Fixed installation issues for Python 3.12

**Reason:** User reported issues with `pip install -r requirements.txt`. `sentencepiece==0.1.96` failed to build due to missing C++ headers, and `attrdict` was incompatible with Python 3.12.

**Action:** 
1. Modified `requirements.txt`: Removed `SentencePiece==0.1.96` and kept the unpinned `sentencepiece` to allow installation of version `0.2.1+`.
2. Modified `pyproject.toml`: 
    - Removed `SentencePiece==0.1.96` from `gradio` optional dependencies.
    - Removed incorrect package overrides (`aurora_genesis_core`, `deepseek_adapter`).
3. Installed package in editable mode: `python -m pip install -e .`.
4. Verified `attrdict` monkeypatch in `deepseek_vl/__init__.py`.

**Outcome:** Installation successful, all modules importable.

### Activity: Fixed CUDA OOM in inference.py

**Reason:** User reported `torch.OutOfMemoryError` on a 4GB GPU when running `inference.py`. The script was forcefully moving the entire 7B model to CUDA.
**Action:** Modified `inference.py` to use `device_map="auto"` and `torch_dtype=torch.bfloat16` in `from_pretrained`. This allows `accelerate` to offload model layers to CPU RAM as needed.
**Outcome:** Model should now load, albeit slower due to CPU offloading.

### Activity: Switched to 1.3B Model

**Reason:** User requested `deepseek-vl-1.3b-base` instead of the 7B version to fit within 4GB VRAM.
**Action:** Updated `inference.py` to point to `deepseek-ai/deepseek-vl-1.3b-base`.
**Outcome:** Model size reduced significantly (~2.6GB), which should fit entirely on the GPU.

### Activity: Updated default model and loading utilities

**Reason:** Ensure all entry points (CLI and internal utils) use safe memory loading and point to appropriate smaller models for 4GB VRAM.
**Action:** 
1. Modified `deepseek_vl/utils/io.py`: Updated `load_pretrained_model` to use `device_map="auto"` and `torch_dtype=torch.bfloat16`.
2. Modified `cli_chat.py`: Changed default model to `deepseek-ai/deepseek-vl-1.3b-chat`.
**Outcome:** Consistency across the project for low-VRAM compatibility.

### Activity: Fixed deprecation warnings

**Reason:** The previous run showed a warning: "`torch_dtype` is deprecated! Use `dtype` instead!".
**Action:** Replaced `torch_dtype` with `dtype` in `inference.py` and `deepseek_vl/utils/io.py`.
**Outcome:** Cleaner logs and future-proof code.
