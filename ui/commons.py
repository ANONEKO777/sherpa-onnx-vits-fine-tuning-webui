import gradio_i18n
from contextlib import contextmanager

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