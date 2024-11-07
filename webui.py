# Description: 建立功能的主頁面
# 標準函式庫
import os
import argparse
# 第三方函式庫
import gradio as gr
from gradio_i18n import gettext, translate_blocks
# 本地函式庫
import vits_fast_fine_tuning.utils as utils
import ui

_theme = gr.themes.Soft()

# 定義整個Gradio介面
def create_interface():
    with gr.Blocks(theme=_theme) as demo:
        lang = gr.Radio(choices=[("English", "en"), ("中文", "zh"), ("日本語", "ja")], label=gettext("Language"))
        
        ui.create_export_onnx_interface(lang=lang)
        ui.create_onnx_inference_interface(lang=lang)

    return demo

# 啟動Gradio介面
if __name__ == "__main__":
    demo = create_interface()
    demo.launch()