# Application Workflow Diagram - DeepSeek-VL

## Initialization Sequence
- `deepseek_vl/__init__.py`:
    - Checks `sys.version_info`.
    - If version >= 3.10:
        - Patches `collections` module by mapping `collections.abc` types to it (fixing `attrdict` compatibility).

## Inference Sequence
- `cli_chat.py` or `inference.py`:
    - Entry point for user.
- `deepseek_vl/models/processing_vlm.py` (`VLChatProcessor`):
    - Prepares multimodal inputs (images + text).
- `deepseek_vl/models/modeling_vlm.py` (`MultiModalityCausalLM`):
    - `prepare_inputs_embeds`: Combines vision and language embeddings.
    - `language_model.generate`: Standard LLM generation loop.
- `deepseek_vl/models/projector.py`:
    - Maps vision features to language space using MLP.
