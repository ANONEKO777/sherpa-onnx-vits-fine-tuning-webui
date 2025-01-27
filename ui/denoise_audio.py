
import os

import gradio as gr
from gradio_i18n import translate_blocks

from .commons import add_prefix_to_translations, gettext, set_page_prefix
from vits_fast_fine_tuning.scripts.denoise_audio import denoise_audio

_translations = {
    "en": {
        "Title": "<h1 style='text-align: center;'>Denoise and Resample Page</h1>",
        "RawAudioDirectory": "Raw Audio Directory",
        "DenoisedAudioDirectory": "Denoised Audio Directory",
        "TargetSampleRate": "Target Sample Rate",
        "DenoiseAudio": "Denoise and Resample Audio",
        "DenoiseResult": "Denoise and Resample Result",
        "DenoiseComplete": "Denoise and Resample complete",
    },
    "zh": {
        "Title": "<h1 style='text-align: center;'>去除雜音和重新採樣頁面</h1>",
        "RawAudioDirectory": "原始音訊目錄",
        "DenoisedAudioDirectory": "去除雜音音訊目錄",
        "TargetSampleRate": "目標取樣率",
        "DenoiseAudio": "去除雜音和重新採樣",
        "DenoiseResult": "去除雜音和重新採樣結果",
        "DenoiseComplete": "去除雜音和重新採樣完成",
    },
    "ja": {
        "Title": "<h1 style='text-align: center;'>ノイズ除去とリサンプリングページ</h1>",
        "RawAudioDirectory": "生オーディオディレクトリ",
        "DenoisedAudioDirectory": "ノイズ除去オーディオディレクトリ",
        "TargetSampleRate": "ターゲットサンプリングレート",
        "DenoiseAudio": "ノイズ除去とリサンプリング",
        "DenoiseResult": "ノイズ除去とリサンプリング結果",
        "DenoiseComplete": "ノイズ除去とリサンプリング完了",
    },
}

_page_prefix = "denoise_audio"
_translations = add_prefix_to_translations(_translations, _page_prefix)

def denoise_audio_event(raw_audio_dir, denoise_audio_dir, target_sr):
    with set_page_prefix(_page_prefix):
        denoise_audio(raw_audio_dir, denoise_audio_dir, target_sr)
        return gettext("DenoiseComplete")

def create_denoise_audio_interface(lang):
    with set_page_prefix(_page_prefix):
        with gr.Blocks() as denoise_audio_blocks:
            title = gr.Markdown(gettext("Title"))

            with gr.Row():
                raw_audio_dir_input = gr.Textbox(label=gettext("RawAudioDirectory"), value="training_data/raw_audio")
                denoise_audio_dir_input = gr.Textbox(label=gettext("DenoisedAudioDirectory"), value="training_data/denoised_audio")

            target_sr = gr.Textbox(label=gettext("TargetSampleRate"), value="16000")
            denoise_button = gr.Button(value=gettext("DenoiseAudio"), variant="primary")
            denoise_result = gr.Textbox(label=gettext("DenoiseResult"))

            denoise_button.click(fn=denoise_audio_event, inputs=[raw_audio_dir_input, denoise_audio_dir_input, target_sr], outputs=denoise_result)

    translate_blocks(block=denoise_audio_blocks, translation=_translations, lang=lang)