# Historical references: read-only, not dependencies

- `suwako-art.zip`: original CUDA-era Suwako CLI and Streamlit app. Its imported `mew.ai.DiffusionService` is not included.
- `desktop.py`: user's older Streamlit-to-pywebview wrapper.
- `st_image_gallery.py` / `st_image_gallery_fancy.py`: original small gallery experiments.
- `labs/ai/booru_test.py`: booru and captioning research. **Do not run the crawler automatically**; third-party terms, copyright, rate limiting, and error handling require deliberate work.
- `labs/ai/gradio_test.py`, `labs/ai/gradio_caption_app.py`: Gradio interface/captioning experiments.
- `labs/ai/llm_chat_test.py`, `labs/ai/chatbot_test.py`: language-model experimentation; not production provider implementations.
- `labs/ui/st_webview.py`: older embedded UI experiment.

These scripts are inspirations rather than supported modules. Some need additional user-created utilities or historical dependency versions. They may contain defects; review before reuse.

The full `labs.zip` archive was **not included** because `labs/web/discord_bot.py` appears to contain a hard-coded Discord bot token. If the credential was real and remains active, rotate it. Never add the original full labs archive to a public repository.
