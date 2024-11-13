import gradio as gr
from gradio_i18n import gettext, translate_blocks

from .commons import add_prefix_to_translations, gettext, set_page_prefix
import vits_fast_fine_tuning.scripts.long_audio_transcribe as long_audio

_translations = {
    "en": {
        "title": "<h1 style='text-align: center;'>Audio Transcription for Training Text Page</h1>",
        "language_model": "Select the language of the audio, which will affect the type of the model after training",
        "whisper_size": "Select the Whisper model size, the larger the size, the higher the accuracy",
        "denoise_audio_dir": "Source audio directory (denoised audio directory)",
        "submit": "Submit",
        "output": "Output",
        "transcribe complete": "Transcribe complete",
    },
    "zh": {
        "title": "<h1 style='text-align: center;'>音訊標注為訓練用文本頁面</h1>",
        "language_model": "選擇音訊的語言，將影響訓練後模型類型",
        "whisper_size": "選擇Whisper模型大小，越大精準度越高",
        "denoise_audio_dir": "來源音訊目錄(去噪音訊目錄)",
        "submit": "提交",
        "output": "輸出結果",
        "transcribe complete": "標注完成",
    },
    "ja": {
        "title": "<h1 style='text-align: center;'>トレーニングテキスト用のオーディオ転写ページ</h1>",
        "language_model": "音声の言語を選択し、トレーニング後のモデルタイプに影響します",
        "whisper_size": "Whisperモデルサイズを選択します。サイズが大きいほど精度が高くなります",
        "denoise_audio_dir": "ソースオーディオディレクトリ（ノイズ除去オーディオディレクトリ）",
        "submit": "提出",
        "output": "出力",
        "transcribe complete": "転写完了",
    },
}

_page_prefix = "transcribe_audio"
_translations = add_prefix_to_translations(_translations, _page_prefix)

def submit(denoise_audio_dir, language_model, whisper_size):
    with set_page_prefix(_page_prefix):
        long_audio.transcribe_audio(denoise_audio_dir, language_model, whisper_size)
        return gettext("transcribe complete")

def create_transcribe_audio_interface(lang):
    with set_page_prefix(_page_prefix):
        with gr.Blocks() as transcribe_audio_blocks:
            title = gr.Markdown(gettext("title"))
            language_model = gr.Dropdown(
                label=gettext("language_model"),
                choices=[("Chinese", "C"), ("Chinese+Japanese", "CJ"), ("Chinese+Japanese+English", "CJE")],
                value="C"
            )
            whisper_size = gr.Dropdown(
                label=gettext("whisper_size"),
                choices=[
                    ("tiny (39 M)", "tiny"),
                    ("base (74 M)", "base"),
                    ("small (244 M)", "small"),
                    ("medium (769 M)", "medium"),
                    ("large (1550 M)", "large"),
                    ("large-v2 (1550 M)", "large-v2")
                ],
                value="large-v2"
            )
            denoise_audio_dir = gr.Textbox(label=gettext("denoise_audio_dir"), value="training_data/denoised_audio/")
            submit_button = gr.Button(gettext("submit"), variant="primary")
            output = gr.Textbox(label=gettext("output"))

            submit_button.click(submit, inputs=[denoise_audio_dir, language_model, whisper_size], outputs=[output])

    translate_blocks(block=transcribe_audio_blocks, translation=_translations, lang=lang)