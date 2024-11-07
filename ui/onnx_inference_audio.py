import os
import argparse
# 第三方函式庫
import gradio as gr
import gradio_i18n
from gradio_i18n import translate_blocks
# 本地函式庫
import onnx_inference
from .commons import add_prefix_to_translations, gettext, set_page_prefix

_onnx_export_folder = "onnx-output"

_translations = {
    "en": {
        "title": "<h1 style='text-align: center;'>Onnx Inference Audio Page</h1>",
        "text": "Speaking text",
        "example_text": "Good morning everyone, welcome to the world of text-to-speech!",
        "sid": "Speaker ID",
        "model_folder": "Select Onnx Model Folder",
        "submit": "Submit",
        "output_audio": "Output Audio",
    },
    "zh": {
        "title": "<h1 style='text-align: center;'>Onnx推理音訊頁面</h1>",
        "text": "說話文字",
        "example_text": "大家早安，歡迎來到文字轉語音的世界！",
        "sid": "語者ID",
        "model_folder": "選擇Onnx模型資料夾",
        "submit": "提交",
        "output_audio": "輸出音訊",
    },
    "ja": {
        "title": "<h1 style='text-align: center;'>Onnx推論オーディオページ</h1>",
        "text": "話者テキスト",
        "example_text": "みなさん、おはようございます。テキスト読み上げの世界へようこそ！",
        "sid": "話者ID",
        "model_folder": "Onnxモデルフォルダを選択",
        "submit": "提出",
        "output_audio": "出力音声",
    },
}

_page_prefix = "onnx_inference"
# 使用函數為每個鍵增加前綴，原因是目前i18n套件的字典是全介面共用的，因此用前綴區分不同功能頁面的翻譯
_translations = add_prefix_to_translations(_translations, _page_prefix)

def get_model_folders():
    return [f.name for f in os.scandir(_onnx_export_folder) if f.is_dir()]

def refresh_model_folders():
    model_folders = get_model_folders()
    return gr.update(choices=model_folders)

def submit(text, sid, model_folder):
    args = argparse.Namespace(
        text=text,
        checkpoint=os.path.join(_onnx_export_folder, model_folder, "model.onnx"),
        sid=sid,
        lexicon=os.path.join(_onnx_export_folder, model_folder, "lexicon.txt"),
        tokens=os.path.join(_onnx_export_folder, model_folder, "tokens.txt"),
    )
    onnx_inference.check_args(args)
    return onnx_inference.infer_by_args(args)

def create_onnx_inference_interface(lang):
    model_folders = get_model_folders()
    with set_page_prefix(_page_prefix):
        with gr.Blocks() as onnx_inference_blocks:
            title = gr.Markdown(gettext("title"))
            text = gr.Textbox(label=gettext("text"), value=gettext("example_text"))
            sid = gr.Number(label=gettext("sid"), value=0, precision=0)
            
            with gr.Row():
                model_folder = gr.Dropdown(label=gettext("model_folder"), choices=model_folders)
                refresh_btn = gr.Button(value="\U0001F504")
                refresh_btn.click(refresh_model_folders, inputs=[], outputs=model_folder)

            submit_button = gr.Button(gettext("submit"))
            output_audio = gr.Audio(label=gettext("output_audio"))

            submit_button.click(submit, inputs=[text, sid, model_folder], outputs=output_audio)
    
    translate_blocks(block=onnx_inference_blocks, translation=_translations, lang=lang)