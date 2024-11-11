
import os

import gradio as gr
from gradio_i18n import translate_blocks

from .commons import add_prefix_to_translations, gettext, set_page_prefix
from vits_fast_fine_tuning.scripts.denoise_audio import denoise_audio

_translations = {
    "en": {
        "title": "<h1 style='text-align: center;'>Denoise and Resample Page</h1>",
        "Raw Audio Directory": "Raw Audio Directory",
        "Denoised Audio Directory": "Denoised Audio Directory",
        "Denoise Audio": "Denoise and Resample Audio",
        "Denoise Result": "Denoise and Resample Result",
        "Denoise complete": "Denoise and Resample complete",
    },
    "zh": {
        "title": "<h1 style='text-align: center;'>去除雜音和重新採樣頁面</h1>",
        "Raw Audio Directory": "原始音訊目錄",
        "Denoised Audio Directory": "去除雜音音訊目錄",
        "Denoise Audio": "去除雜音和重新採樣",
        "Denoise Result": "去除雜音和重新採樣結果",
        "Denoise complete": "去除雜音和重新採樣完成",
    },
    "ja": {
        "title": "<h1 style='text-align: center;'>ノイズ除去とリサンプリングページ</h1>",
        "Raw Audio Directory": "生オーディオディレクトリ",
        "Denoised Audio Directory": "ノイズ除去オーディオディレクトリ",
        "Denoise Audio": "ノイズ除去とリサンプリング",
        "Denoise Result": "ノイズ除去とリサンプリング結果",
        "Denoise complete": "ノイズ除去とリサンプリング完了",
    },
}

_page_prefix = "denoise_audio"
_translations = add_prefix_to_translations(_translations, _page_prefix)

def denoise_audio_event(raw_audio_dir, denoise_audio_dir):
    with set_page_prefix(_page_prefix):
        denoise_audio(raw_audio_dir, denoise_audio_dir)
        return gettext("Denoise complete")

def create_denoise_audio_interface(lang):
    with set_page_prefix(_page_prefix):
        with gr.Blocks() as denoise_audio_blocks:
            title = gr.Markdown(gettext("title"))

            with gr.Row():
                raw_audio_dir_input = gr.Textbox(label=gettext("Raw Audio Directory"), value="training_data/raw_audio")
                denoise_audio_dir_input = gr.Textbox(label=gettext("Denoised Audio Directory"), value="training_data/denoised_audio")

            denoise_button = gr.Button(value=gettext("Denoise Audio"), variant="primary")
            denoise_result = gr.Textbox(label=gettext("Denoise Result"))

            denoise_button.click(fn=denoise_audio_event, inputs=[raw_audio_dir_input, denoise_audio_dir_input], outputs=denoise_result)
    
    translate_blocks(block=denoise_audio_blocks, translation=_translations, lang=lang)