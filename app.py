!pip install gradio diffusers transformers accelerate huggingface_hub spaces
import os; os.environ["HF_TOKEN"] = "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"

import spaces
import gradio as gr
import os
from huggingface_hub import InferenceClient

HF_TOKEN = os.environ.get("HF_TOKEN", "")

MODELS = {
    "Lustly AI Flux (Uncensored)": "lustlyai/Flux_Lustly.ai_Uncensored_nsfw_v1",
    "Heartsync Flux (Uncensored)": "Heartsync/Flux-NSFW-uncensored",
    "FLUX Uncensored Merged": "shauray/FLUX-UNCENSORED-merged",
    "UnfilteredAI NSFW Gen V2": "UnfilteredAI/NSFW-gen-v2",
    "FLUX.1 Schnell (Fast/Permissive)": "black-forest-labs/FLUX.1-schnell",
    "FHDR Uncensored": "kpsss34/FHDR_Uncensored",
    "Z-Image Turbo NSFW": "thutes-gbr25/NSFW-MASTER-Z-IMAGE-TURBO",
    "Illustrious SDXL NSFW": "John6666/wai-nsfw-illustrious-v80-sdxl",
    "Qwen Image NSFW": "starsfriday/Qwen-Image-NSFW",
    "UnfilteredAI Anime NSFW": "UnfilteredAI/NSFW-GEN-ANIME",
}

def generate_image(prompt, model_name, negative_prompt, steps, guidance):
    if not prompt.strip():
        return None, "⚠️ Please enter a prompt."
    if not HF_TOKEN:
        return None, "❌ HF_TOKEN secret is not set."

    model_id = MODELS[model_name]

    try:
        client = InferenceClient(token=HF_TOKEN)
        image = client.text_to_image(
            prompt=prompt,
            model=model_id,
            negative_prompt=negative_prompt.strip() if negative_prompt.strip() else None,
            num_inference_steps=int(steps),
            guidance_scale=float(guidance),
        )
        return image, "✅ Image generated!"
    except Exception as e:
        error = str(e)
        if "503" in error or "loading" in error.lower():
            return None, "⏳ Model is loading, please wait 30 seconds and try again."
        elif "not supported" in error.lower() or "404" in error:
            return None, "❌ This model is not available on HF Inference API. It may require a paid endpoint."
        return None, f"❌ {error[:400]}"

with gr.Blocks(title="Sila Image Studio", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎨 Sila Image Studio")
    gr.Markdown("Your private AI image generator")

    with gr.Row():
        with gr.Column(scale=1):
            model_selector = gr.Dropdown(
                choices=list(MODELS.keys()),
                value=list(MODELS.keys())[0],
                label="🤖 Model"
            )
            prompt = gr.Textbox(
                label="✏️ Prompt",
                placeholder="Describe the image you want to create...",
                lines=4
            )
            negative_prompt = gr.Textbox(
                label="🚫 Negative Prompt (optional)",
                placeholder="What to avoid in the image...",
                lines=2
            )
            with gr.Row():
                steps = gr.Slider(minimum=4, maximum=50, value=28, step=1, label="Steps")
                guidance = gr.Slider(minimum=1.0, maximum=15.0, value=3.5, step=0.5, label="Guidance")
            generate_btn = gr.Button("✨ Generate", variant="primary", size="lg")
            status = gr.Textbox(label="Status", interactive=False)

        with gr.Column(scale=1):
            output_image = gr.Image(label="🖼️ Generated Image", height=512)

    generate_btn.click(
        fn=generate_image,
        inputs=[prompt, model_selector, negative_prompt, steps, guidance],
        outputs=[output_image, status]
    )

demo.launch()
