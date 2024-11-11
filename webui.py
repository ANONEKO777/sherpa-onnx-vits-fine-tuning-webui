# Description: 建立功能的主頁面
from typing import Union, Iterable
# 第三方函式庫
import gradio as gr
from gradio.themes.soft import Soft
from gradio.themes.utils import colors, fonts, sizes
from gradio_i18n import gettext
# 本地函式庫
import vits_fast_fine_tuning.utils as utils
import ui

# _theme = gr.themes.Soft()
class MyCustomTheme(Soft):
    def __init__(
        self,
        *,
        primary_hue: Union[colors.Color, str] = colors.blue,
        secondary_hue: Union[colors.Color, str] = colors.blue,
        neutral_hue: Union[colors.Color, str] = colors.gray,
        spacing_size: Union[sizes.Size, str] = sizes.spacing_md,
        radius_size: Union[sizes.Size, str] = sizes.radius_md,
        text_size: Union[sizes.Size, str] = sizes.text_lg,
        font: Union[fonts.Font, str, Iterable[Union[fonts.Font, str]]] = (
            fonts.GoogleFont("Quicksand"),
            "ui-sans-serif",
            "sans-serif",
        ),
        font_mono: Union[fonts.Font, str, Iterable[Union[fonts.Font, str]]] = (
            fonts.GoogleFont("IBM Plex Mono"),
            "ui-monospace",
            "monospace",
        ),
    ):
        super().__init__(
            primary_hue=primary_hue,
            secondary_hue=secondary_hue,
            neutral_hue=neutral_hue,
            spacing_size=spacing_size,
            radius_size=radius_size,
            text_size=text_size,
            font=font,
            font_mono=font_mono,
        )
        super().set(
            block_border_width="1px",
            block_border_width_dark="1px",
            border_color_primary=colors.blue.c200,
            border_color_primary_dark=colors.pink.c700,
            # Cancel Button
            button_cancel_background_fill=colors.pink.c200,
            button_cancel_background_fill_dark=colors.pink.c700,
            button_cancel_background_fill_hover=colors.pink.c100,
            button_cancel_background_fill_hover_dark=colors.pink.c600,
            button_cancel_border_color=colors.pink.c200,
            button_cancel_border_color_dark=colors.pink.c600,
            button_cancel_border_color_hover=colors.pink.c200,
            button_cancel_border_color_hover_dark=colors.pink.c600,
            button_cancel_text_color=colors.pink.c700,
            button_cancel_text_color_dark="white",
        )

# 定義整個Gradio介面
def create_interface():
    with gr.Blocks(theme=MyCustomTheme()) as demo:
        lang = gr.Radio(choices=[("English", "en"), ("中文", "zh"), ("日本語", "ja")], label=gettext("Language"))
        
        ui.create_youtube_download_interface(lang=lang)
        ui.create_export_onnx_interface(lang=lang)
        ui.create_onnx_inference_interface(lang=lang)

    return demo

# 啟動Gradio介面
if __name__ == "__main__":
    demo = create_interface()
    demo.launch()