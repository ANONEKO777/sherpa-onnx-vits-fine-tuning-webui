import gradio as gr
from gradio_i18n import gettext, translate_blocks

from .commons import add_prefix_to_translations, gettext, set_page_prefix
from vits_fast_fine_tuning.preprocess_v2 import preprocess

_translations = {
    "en": {
        "title": "<h1 style='text-align: center;'>Preprocess Data for Fine-Tuning</h1>",
        "languages": "Select languages",
        "hint": "If you have less than 100 training audio samples, it is recommended to check this box. Note that auxiliary data will include Chinese, Japanese, and English languages.",
        "add_auxiliary_data": "Add auxiliary data",
        "submit": "Submit",
        "output": "Output",
        "preprocess complete": "Preprocess complete",
    },
    "zh": {
        "title": "<h1 style='text-align: center;'>預處理資料以進行微調</h1>",
        "languages": "選擇語言",
        "hint": "如果您的訓練音訊樣本少於100個，建議勾選此框。請注意，輔助資料將包括中文、日文和英文語言。",
        "add_auxiliary_data": "添加輔助資料",
        "submit": "提交",
        "output": "輸出結果",
        "preprocess complete": "預處理完成",
    },
    "ja": {
        "title": "<h1 style='text-align: center;'>微調用のデータを前処理する</h1>",
        "languages": "言語を選択",
        "hint": "トレーニングオーディオサンプルが100未満の場合は、このボックスをチェックすることをお勧めします。補助データには、中国語、日本語、英語の言語が含まれます。",
        "add_auxiliary_data": "補助データを追加",
        "submit": "提出",
        "output": "出力",
        "preprocess complete": "前処理完了",
    },
}

_page_prefix = "preprocess"
_translations = add_prefix_to_translations(_translations, _page_prefix)

def submit(languages, add_auxiliary_data):
    with set_page_prefix(_page_prefix):
        preprocess(languages, add_auxiliary_data)
        return gettext("preprocess complete")

def create_preprocess_interface(lang):
    with set_page_prefix(_page_prefix):
        with gr.Blocks() as preprocess_blocks:
            title = gr.Markdown(gettext("title"))
            languages = gr.Dropdown(
                label=gettext("languages"),
                choices=[("Chinese", "C"), ("Chinese+Japanese", "CJ"), ("Chinese+Japanese+English", "CJE")],
                value="C"
            )
            add_auxiliary_data = gr.Checkbox(label=gettext("add_auxiliary_data"), info=gettext("hint"), value=False)
            submit_button = gr.Button(gettext("submit"), variant="primary")
            output = gr.Textbox(label=gettext("output"))

            submit_button.click(submit, inputs=[languages, add_auxiliary_data], outputs=[output])

    translate_blocks(block=preprocess_blocks, translation=_translations, lang=lang)