import random
import numpy as np
import torch
from chatterbox.src.chatterbox.tts import ChatterboxTTS
import gradio as gr
import spaces

MODEL = None

DEFAULT_CONFIG = {
    "audio": 'https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/hi_f1.flac',
    "text": 'नमस्ते, आप कैसे हैं? आज मौसम बहुत सुहावना है।',
}

EXAMPLES = [
        ['नमस्ते, आप कैसे हैं? आज मौसम बहुत सुहावना है।', 'https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/hi_f1.flac', 0.5, 0.8, 0, 0.5],
        ['भारत अपनी विविधता, संस्कृति और स्वादिष्ट खाने के लिए प्रसिद्ध है।', 'https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/hi_f1.flac', 0.5, 0.8, 0, 0.5],
        ['क्या आप एक कप चाय और गरमागरम समोसे का आनंद लेना चाहेंगे?', 'https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/hi_f1.flac', 0.5, 0.8, 0, 0.5]
]


def default_audio_for_ui():
    return DEFAULT_CONFIG.get("audio")


def default_text_for_ui():
    return DEFAULT_CONFIG.get("text", "")


def get_or_load_model():
    global MODEL
    if MODEL is None:
        print("Model not loaded, initializing on CPU...")
        MODEL = ChatterboxTTS.from_pretrained("cpu")
        print("Model loaded.")
    return MODEL


def set_seed(seed: int, device: str):
    torch.manual_seed(seed)
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)


@spaces.GPU
def generate_tts_audio(
    text_input: str,
    audio_prompt_path_input: str = None,
    exaggeration_input: float = 0.5,
    temperature_input: float = 0.8,
    seed_num_input: int = 0,
    cfgw_input: float = 0.5,
):
    """Generate speech from text with optional reference audio styling."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    current_model = get_or_load_model()
    current_model.to(device)
    if seed_num_input != 0:
        set_seed(int(seed_num_input), device)
    print(f"Generating on {device} for text: '{text_input[:50]}...'")
    chosen_prompt = audio_prompt_path_input or default_audio_for_ui()
    generate_kwargs = {
        "exaggeration": exaggeration_input,
        "temperature": temperature_input,
        "cfg_weight": cfgw_input,
    }
    if chosen_prompt:
        generate_kwargs["audio_prompt_path"] = chosen_prompt
        print(f"Using audio prompt: {chosen_prompt}")
    wav = current_model.generate(text_input[:300], **generate_kwargs)
    return (current_model.sr, wav.squeeze(0).cpu().numpy())


with gr.Blocks() as demo:
    gr.Markdown(
        """
        # Chatterbox Multilingual TTS — Hindi
        Chatterbox TTS fine-tuned for Hindi (hi).
        Powered by model [`ResembleAI/Chatterbox-Multilingual-hi`](https://huggingface.co/ResembleAI/Chatterbox-Multilingual-hi).
        """
    )
    with gr.Row():
        with gr.Column():
            text = gr.Textbox(
                value=default_text_for_ui(),
                label="Text to synthesize (max chars 300)",
                max_lines=5,
            )
            ref_wav = gr.Audio(
                sources=["upload", "microphone"],
                type="filepath",
                label="Reference Audio File (Optional)",
                value=default_audio_for_ui(),
            )
            exaggeration = gr.Slider(0.25, 2, step=.05, label="Exaggeration (Neutral = 0.5)", value=.5)
            cfg_weight = gr.Slider(0.2, 1, step=.05, label="CFG/Pace", value=0.5)
            with gr.Accordion("More options", open=False):
                seed_num = gr.Number(value=0, label="Random seed (0 for random)")
                temp = gr.Slider(0.05, 5, step=.05, label="Temperature", value=.8)
            run_btn = gr.Button("Generate", variant="primary")
        with gr.Column():
            audio_output = gr.Audio(label="Output Audio")

    inputs = [text, ref_wav, exaggeration, temp, seed_num, cfg_weight]
    run_btn.click(fn=generate_tts_audio, inputs=inputs, outputs=[audio_output])

    gr.Examples(
        examples=EXAMPLES,
        inputs=[text, ref_wav, exaggeration, temp, seed_num, cfg_weight],
        label="Examples",
    )

demo.launch(mcp_server=True)
