import gradio as gr
from gradio_i18n import gettext, translate_blocks

from gradio_log import Log
from .commons import add_prefix_to_translations, gettext, set_page_prefix

_translations = {
    "en": {
        "Console": "View messages in the console",
        "Console Output": "Console Output",
    },
    "zh": {
        "Console": "查看控制台的訊息",
        "Console Output": "控制台訊息",
    },
    "ja": {
        "Console": "コンソールのメッセージを表示",
        "Console Output": "コンソール出力",
    },
}

_page_prefix = "stdout"
_translations = add_prefix_to_translations(_translations, _page_prefix)

def create_stdout_interface(lang):
    with set_page_prefix(_page_prefix):
        with gr.Blocks() as stdout_blocks:
            # 使用浮動式視窗
            with gr.Accordion(label=gettext("Console"), open=False, elem_id="floating-window"):
                Log("./sys.stdout.log", dark=True, xterm_font_weight="bold", height=500)

    translate_blocks(block=stdout_blocks, translation=_translations, lang=lang)
