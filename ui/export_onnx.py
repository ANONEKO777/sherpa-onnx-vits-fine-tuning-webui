# Description: 透過 Gradio 建立 Pytorch 模型轉出 Sherpa-Onnx 模型的設定頁面
import os
# 第三方函式庫
import gradio as gr
from gradio_i18n import gettext, translate_blocks
# 本地函式庫
import export_vits_fast_fine_tuning_onnx as export_onnx

_theme = gr.themes.Soft()
_pytorch_model_folder = "models"
_onnx_export_folder = "onnx-output"

# 定義多國語言文字
_translations = {
    "en": {
        "title": "<h1 style='text-align: center;'>Pytorch Model to Sherpa-Onnx Model Configuration Page</h1>",
        "model_folder": "Select Pytorch Model Folder",
        "pytorch_file": "Select Pytorch Model File",
        "model_comment": "Enter Model Comment",
        "language": "Select Language",
        "model_name": "Enter Model Name",
        "submit": "Submit",
        "output": "Output",
        "config: {config}\n": "config: {config}\n",
        "checkpoint: {checkpoint}\n": "checkpoint: {checkpoint}\n",
        "output_dir: {output_dir}\n": "output_dir: {output_dir}\n",
        "comment: {comment}\n": "comment: {comment}\n",
        "language: {language}\n": "language: {language}\n",
    },
    "zh": {
        "title": "<h1 style='text-align: center;'>Pytorch模型轉出Sherpa-Onnx模型設定頁面</h1>",
        "model_folder": "選擇Pytorch模型位置",
        "pytorch_file": "選擇Pytorch模型檔案",
        "model_comment": "輸入模型註解",
        "language": "選擇語言",
        "model_name": "輸入模型名稱",
        "submit": "提交",
        "output": "輸出結果",
        "config: {config}\n": "設定檔: {config}\n",
        "checkpoint: {checkpoint}\n": "pth模型: {checkpoint}\n",
        "output_dir: {output_dir}\n": "輸出目錄: {output_dir}\n",
        "comment: {comment}\n": "註解: {comment}\n",
        "language: {language}\n": "語言: {language}\n",
    },
    "ja": {
        "title": "<h1 style='text-align: center;'>PytorchモデルをSherpa-Onnxモデルに変換する設定ページ</h1>",
        "model_folder": "Pytorchモデルの場所を選択",
        "pytorch_file": "Pytorchモデルファイルを選択",
        "model_comment": "モデルコメントを入力",
        "language": "言語を選択",
        "model_name": "モデル名を入力",
        "submit": "送信",
        "output": "出力",
        "config: {config}\n": "設定: {config}\n",
        "checkpoint: {checkpoint}\n": "pthモデル: {checkpoint}\n",
        "output_dir: {output_dir}\n": "出力ディレクトリ: {output_dir}\n",
        "comment: {comment}\n": "コメント: {comment}\n",
        "language: {language}\n": "言語: {language}\n",
    }
}

# 讀取 models 資料夾底下的子資料夾項目
def get_model_folders():
    return [f.name for f in os.scandir(_pytorch_model_folder) if f.is_dir()]

# 初始化 model_folders
_model_folders = get_model_folders()

# 重新整理 model_folders
def refresh_model_folders():
    global _model_folders
    _model_folders = get_model_folders()
    return gr.update(choices=_model_folders)

# 定義更新 pytorch_file 選項的函數
def update_pytorch_files(model_folder):
    # 獲取選定文件夾中的所有 .pth 文件
    pth_files = [f for f in os.listdir(os.path.join(_pytorch_model_folder, model_folder)) if f.endswith('.pth')]
    return gr.update(choices=pth_files)


def export_submit(model_folder, model_comment, language, model_name, pytorch_file):
    args = argparse.Namespace(
        config=os.path.join(_pytorch_model_folder, model_folder, "config.json"),
        checkpoint=os.path.join(_pytorch_model_folder, model_folder, pytorch_file),
        output_dir=_onnx_export_folder,
        comment=model_comment,
        language=language,
        model_name=f"vits-{model_name}",
    )
    hps = utils.get_hparams_from_file(args.config)
    export_onnx.export_onnx_model(hps, args)
    # 返回args所有參數的字串
    str1 = gettext("config: {config}\n").format(config=args.config)
    str2 = gettext("checkpoint: {checkpoint}\n").format(checkpoint=args.checkpoint)
    str3 = gettext("output_dir: {output_dir}\n").format(output_dir=args.output_dir)
    str4 = gettext("comment: {comment}\n").format(comment=args.comment)
    str5 = gettext("language: {language}\n").format(language=args.language)

    return str1 + str2 + str3 + str4 + str5

def create_export_onnx_interface(lang):
    with gr.Row():
        model_folder = gr.Dropdown(label=gettext("model_folder"), choices=_model_folders)
        refresh_model_folder_btn = gr.Button(value="\U0001F504")
        # 按下重新整理按鈕時，重新讀取 model_folder 內的資料夾
        refresh_model_folder_btn.click(refresh_model_folders, outputs=model_folder)

    pytorch_file = gr.Dropdown(label=gettext("pytorch_file"), choices=[])
    # 當 model_folder 選擇改變時，更新 pytorch_file 的選項
    model_folder.change(fn=update_pytorch_files, inputs=model_folder, outputs=pytorch_file)
    model_comment = gr.Textbox(label=gettext("model_comment"))
    languages = ["Chinese", "English+Chinese", "English+Chinese+Japanese"]
    language = gr.Dropdown(label=gettext("language"), choices=languages)
    model_name = gr.Textbox(label=gettext("model_name"))
    submit_button = gr.Button(gettext("submit"))
    output = gr.Textbox(label=gettext("output"))

    submit_button.click(export_submit, inputs=[model_folder, model_comment, language, model_name, pytorch_file], outputs=output)
    translate_blocks(translation=_translations, lang=lang)
