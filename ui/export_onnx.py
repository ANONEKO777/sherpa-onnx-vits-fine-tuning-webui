# Description: 透過 Gradio 建立 Pytorch 模型轉出 Sherpa-Onnx 模型的設定頁面
import os
import argparse
# 第三方函式庫
import gradio as gr
from gradio_i18n import gettext, translate_blocks
# 本地函式庫
from vits_fast_fine_tuning import utils
import export_vits_fast_fine_tuning_onnx as export_onnx
from .commons import add_prefix_to_translations, gettext, set_page_prefix
from .commons import get_vits_model_folders, get_vits_model_path, get_onnx_export_path, update_pytorch_files

# 定義多國語言文字
_translations = {
    "en": {
        "title": "<h1 style='text-align: center;'>Pytorch Model to Sherpa-Onnx Model Configuration Page</h1>",
        "model_folder": "Select Pytorch Model Folder",
        "pytorch_file": "Select Pytorch Model File",
        "Pytorch File Info": "Usually select G_latest.pth",
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
        "Pytorch File Info": "通常選擇G_latest.pth",
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
        "Pytorch File Info": "通常はG_latest.pthを選択します",
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

_page_prefix = "export_onnx"
# 使用函數為每個鍵增加前綴，原因是目前i18n套件的字典是全介面共用的，因此用前綴區分不同功能頁面的翻譯
_translations = add_prefix_to_translations(_translations, _page_prefix)

# 重新整理 model_folders
def refresh_model_folders():
    model_folders = get_vits_model_folders()
    return gr.update(choices=model_folders)

def export_submit(model_folder, model_comment, language, model_name, pytorch_file):
    with set_page_prefix(_page_prefix):
        args = argparse.Namespace(
            config=os.path.join(get_vits_model_path(), model_folder, "config.json"),
            checkpoint=os.path.join(get_vits_model_path(), model_folder, pytorch_file),
            output_dir=get_onnx_export_path(),
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
    model_folders = get_vits_model_folders()
    with set_page_prefix(_page_prefix):
        with gr.Blocks() as export_onnx_blocks:
            title = gr.Markdown(gettext("title"))

            with gr.Row():
                model_folder = gr.Dropdown(label=gettext("model_folder"), choices=model_folders)
                refresh_model_folder_btn = gr.Button(value="\U0001F504")
                # 按下重新整理按鈕時，重新讀取 model_folder 內的資料夾
                refresh_model_folder_btn.click(refresh_model_folders, outputs=model_folder)

            pytorch_file = gr.Dropdown(label=gettext("pytorch_file"), choices=[], info=gettext("Pytorch File Info"))
            # 當 model_folder 選擇改變時，更新 pytorch_file 的選項
            model_folder.change(fn=update_pytorch_files, inputs=model_folder, outputs=pytorch_file)
            model_comment = gr.Textbox(label=gettext("model_comment"))
            languages = ["Chinese", "Chinese+Japanese", "Chinese+Japanese+English"]
            language = gr.Dropdown(label=gettext("language"), choices=languages)
            model_name = gr.Textbox(label=gettext("model_name"))
            submit_button = gr.Button(gettext("submit"), variant="primary")
            output = gr.Textbox(label=gettext("output"))

            submit_button.click(export_submit, inputs=[model_folder, model_comment, language, model_name, pytorch_file], outputs=output)

    translate_blocks(block=export_onnx_blocks, translation=_translations, lang=lang)
