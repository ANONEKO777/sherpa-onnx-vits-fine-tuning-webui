import os
import gradio as gr
from gradio_i18n import gettext, translate_blocks

from vits_fast_fine_tuning.VC_inference import create_tts_fn, SynthesizerTrn, utils
from .commons import gettext, set_page_prefix, add_prefix_to_translations, get_vits_model_folders, get_vits_model_path, update_pytorch_files

import torch
device = "cuda:0" if torch.cuda.is_available() else "cpu"

_translations = {
    "en": {
        "Title": "<h1 style='text-align: center;'>VITS PyTorch Inference Page</h1>",
        "Speaker": "Speaker",
        "ConfigPath": "Config Path",
        "Text": "Speaking text",
        "PytorchFile": "Select Pytorch Model File",
        "PytorchFileInfo": "Usually select G_latest.pth",
        "ModelFolder": "Select Pytorch Model Folder",
        "ExampleText": "Good morning everyone, welcome to the world of text to speech!",
        "Language": "Language",
        "LanguageInfo": "Please select your text language, if you choose mix, you need to add [JA] or [EN] or [ZH] before and after the text to distinguish.",
        "Submit": "Submit",
        "OutputAudio": "Output Audio",
    },
    "zh": {
        "Title": "<h1 style='text-align: center;'>VITS PyTorch推理音訊頁面</h1>",
        "Speaker": "語者",
        "ConfigPath": "設定檔路徑",
        "Text": "說話文字",
        "ModelFolder": "選擇Pytorch模型資料夾",
        "PytorchFile": "選擇Pytorch模型檔案",
        "PytorchFileInfo": "通常選擇G_latest.pth",
        "ExampleText": "大家早安，歡迎來到文字轉語音的世界！",
        "Language": "語言",
        "LanguageInfo": "請選擇你的文字語言，若選擇mix，則需要將文字前後加上[JA]或[EN]或[ZH]，以示區隔。",
        "Submit": "提交",
        "OutputAudio": "輸出音訊",
    },
    "ja": {
        "Title": "<h1 style='text-align: center;'>VITS PyTorch推論ページ</h1>",
        "Speaker": "話者",
        "ConfigPath": "設定ファイルのパス",
        "Text": "話者テキスト",
        "ModelFolder": "Pytorchモデルフォルダを選択",
        "PytorchFile": "Pytorchモデルファイルを選択",
        "PytorchFileInfo": "通常はG_latest.pthを選択します",
        "ExampleText": "みなさん、おはようございます。テキスト読み上げの世界へようこそ！",
        "Language": "言語",
        "LanguageInfo": "テキストの言語を選択してください。mixを選択した場合、テキストの前後に[JA]または[EN]または[ZH]を追加して区別する必要があります。",
        "Submit": "提出",
        "OutputAudio": "出力音声",
    },
}

_page_prefix = "vits_pytorch_inference"
# 使用函數為每個鍵增加前綴，原因是目前i18n套件的字典是全介面共用的，因此用前綴區分不同功能頁面的翻譯
_translations = add_prefix_to_translations(_translations, _page_prefix)


# 重新整理 model_folders
def refresh_model_folders():
    model_folders = get_vits_model_folders()
    return gr.update(choices=model_folders)

def get_speakers(config_path):
    hps = utils.get_hparams_from_file(config_path)
    speakers = list(hps.speakers.keys())
    return speakers

def refresh_speakers(config_path):
    return gr.update(choices=get_speakers(config_path))

def run_tts_fn(model_folder, pytorch_file, config_path, text, speaker, language, speed):
    checkpoint = os.path.join(get_vits_model_path(), model_folder, pytorch_file)
    hps = utils.get_hparams_from_file(config_path)
    net_g = SynthesizerTrn(
        len(hps.symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        **hps.model).to(device)
    _ = net_g.eval()
    _ = utils.load_checkpoint(checkpoint, net_g, None)
    speaker_ids = hps.speakers
    # speakers = list(hps.speakers.keys())
    tts_fn = create_tts_fn(net_g, hps, speaker_ids)
    _, audio = tts_fn(text, speaker, language, speed)
    return audio

def create_vits_pytorch_inference_interface(lang):
    model_folders = get_vits_model_folders()
    with set_page_prefix(_page_prefix):
        with gr.Blocks() as vits_pytorch_inference_blocks:
            title = gr.Markdown(gettext("Title"))
            with gr.Row():
                config_path = gr.Textbox(label=gettext("ConfigPath"), value="training_data/configs/modified_finetune_speaker.json")
                refresh_config_btn = gr.Button(value="\U0001F504")

            speaker = gr.Dropdown(label=gettext("Speaker"), choices=get_speakers(config_path.value))
            refresh_config_btn.click(refresh_speakers, inputs=[config_path], outputs=speaker)
            text = gr.Textbox(label=gettext("Text"), value=gettext("ExampleText"))
            language = gr.Dropdown(label=gettext("Language"), choices=['日本語', '正體中文', 'English', 'Mix'], value='正體中文', info=gettext("LanguageInfo"))

            with gr.Row():
                model_folder = gr.Dropdown(label=gettext("ModelFolder"), choices=model_folders)
                refresh_btn = gr.Button(value="\U0001F504")
                refresh_btn.click(refresh_model_folders, inputs=[], outputs=model_folder)

            pytorch_file = gr.Dropdown(label=gettext("PytorchFile"), choices=[], info=gettext("PytorchFileInfo"))
            # 當 model_folder 選擇改變時，更新 pytorch_file 的選項
            model_folder.change(fn=update_pytorch_files, inputs=model_folder, outputs=pytorch_file)

            speed = gr.Slider(label='Speed', minimum=0.1, maximum=5, value=1, step=0.1)
            submit_button = gr.Button(gettext("Submit"), variant="primary")
            output_audio = gr.Audio(label=gettext("OutputAudio"))

            submit_button.click(run_tts_fn, inputs=[model_folder, pytorch_file, config_path, text, speaker, language, speed], outputs=output_audio)

    translate_blocks(block=vits_pytorch_inference_blocks, translation=_translations, lang=lang)
