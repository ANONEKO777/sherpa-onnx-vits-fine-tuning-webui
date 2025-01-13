from typing import Union, Iterable
import os

import gradio
import gradio_i18n
from gradio.themes.soft import Soft
from gradio.themes.utils import colors, fonts, sizes

from contextlib import contextmanager

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

css = """
#floating-window {
    position: fixed;
    bottom: 0;
    right: 20px;
    width: 50%;
    background-color: #bfdbfe;
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    z-index: 1000;
}
"""

_page_prefix = None

@contextmanager
def set_page_prefix(prefix):
    global _page_prefix
    old_prefix = _page_prefix
    _page_prefix = prefix
    try:
        yield
    finally:
        _page_prefix = old_prefix

def add_prefix_to_translations(translations, prefix):
    return {
        lang: {f"{prefix}_{key}": value for key, value in lang_translations.items()}
        for lang, lang_translations in translations.items()
    }

def gettext(key):
    if _page_prefix is None:
        raise ValueError("Page prefix is not set. Please call set_page_prefix() first.")
    return gradio_i18n.gettext(f"{_page_prefix}_{key}")

# 本專案輸出的模型資料夾預設路徑
_pytorch_model_folder = "models"
_onnx_export_folder = "onnx-output"

def get_vits_model_path():
    return _pytorch_model_folder

def get_onnx_export_path():
    return _onnx_export_folder

# 讀取資料夾底下的子資料夾項目
def get_vits_model_folders():
    return [f.name for f in os.scandir(_pytorch_model_folder) if f.is_dir()]

def get_onnx_model_folders():
    return [f.name for f in os.scandir(_onnx_export_folder) if f.is_dir()]

# 會自動搜尋pytorch模型資料夾底下的.pth檔案，並更新給gradio的dropdown
def update_pytorch_files(model_folder):
    # 獲取選定文件夾中的所有 .pth 文件
    pth_files = [f for f in os.listdir(os.path.join(get_vits_model_path(), model_folder)) if f.endswith('.pth')]
    return gradio.update(choices=pth_files)