import os
import gradio as gr
from gradio_i18n import gettext, translate_blocks

theme = gr.themes.Soft()

# 定義多國語言文字
translations = {
    "en": {
        "title": "Model Configuration Page",
        "model_folder": "Select Model Location",
        "model_comment": "Enter Model Comment",
        "language": "Select Language",
        "model_name": "Enter Model Name",
        "submit": "Submit",
        "output": "Output",
        "model folder: {model_folder}\nmodel comment: {model_comment}\nlanguage: {language}\nmodel name: {model_name}": "model folder: {model_folder}\nmodel comment: {model_comment}\nlanguage: {language}\nmodel name: {model_name}",
    },
    "zh": {
        "title": "模型設定頁面",
        "model_folder": "選擇模型位置",
        "model_comment": "輸入模型註解",
        "language": "選擇語言",
        "model_name": "輸入模型名稱",
        "submit": "提交",
        "output": "輸出結果",
        "model folder: {model_folder}\nmodel comment: {model_comment}\nlanguage: {language}\nmodel name: {model_name}": "模型位置: {model_folder}\n模型註解: {model_comment}\n語言: {language}\n模型名稱: {model_name}",
    },
    "ja": {
        "title": "モデル設定ページ",
        "model_folder": "モデルの場所を選択",
        "model_comment": "モデルコメントを入力",
        "language": "言語を選択",
        "model_name": "モデル名を入力",
        "submit": "送信",
        "output": "出力",
        "model folder: {model_folder}\nmodel comment: {model_comment}\nlanguage: {language}\nmodel name: {model_name}": "モデルの場所: {model_folder}\nモデルコメント: {model_comment}\n言語: {language}\nモデル名: {model_name}",
    }
}

# 讀取 models 資料夾底下的子資料夾項目
def get_model_folders():
    model_dir = "models"
    return [f.name for f in os.scandir(model_dir) if f.is_dir()]


def submit(model_folder, model_comment, language, model_name):
    return gettext("model folder: {model_folder}\nmodel comment: {model_comment}\nlanguage: {language}\nmodel name: {model_name}").format(model_folder=model_folder, model_comment=model_comment, language=language, model_name=model_name)


# 定義 Gradio 介面
def create_interface():
    model_folders = get_model_folders()
    languages = ["Chinese", "English+Chinese", "English+Chinese+Japanese"]

    with gr.Blocks(theme=theme) as demo:
        lang = gr.Radio(choices=[("English", "en"), ("中文", "zh"), ("日本語", "ja")], label=gettext("Language"))
        title = gr.Markdown(gettext("title"))
        model_folder = gr.Dropdown(label=gettext("model_folder"), choices=model_folders)
        model_comment = gr.Textbox(label=gettext("model_comment"))
        language = gr.Dropdown(label=gettext("language"), choices=languages)
        model_name = gr.Textbox(label=gettext("model_name"))
        submit_button = gr.Button(gettext("submit"))
        output = gr.Textbox(label=gettext("output"))

        submit_button.click(submit, inputs=[model_folder, model_comment, language, model_name], outputs=output)
        translate_blocks(translation=translations, lang=lang)

    return demo

# 啟動 Gradio 介面
if __name__ == "__main__":
    demo = create_interface()
    demo.launch()