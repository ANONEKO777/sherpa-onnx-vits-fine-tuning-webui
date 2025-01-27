import gradio as gr
from gradio_i18n import gettext, translate_blocks

from .commons import add_prefix_to_translations, gettext, set_page_prefix
import vits_fast_fine_tuning.scripts.long_audio_transcribe as long_audio

_translations = {
    "en": {
        "Title": "<h1 style='text-align: center;'>Audio Transcription for Training Text Page</h1>",
        "LanguageModel": "Select the language of the audio, which will affect the type of the model after training",
        "WhisperSize": "Select the Whisper model size, the larger the size, the higher the accuracy",
        "DenoiseAudioDir": "Source audio directory (denoised audio directory)",
        "TargetSampleRate": "Target Sample Rate",
        "MergeCount": "For fast-speaking voice actors, you can use Merge Count to combine multiple sentences into one to avoid the problem of training too short speech",
        "Submit": "Submit",
        "Output": "Output",
        "TranscribeComplete": "Transcribe complete",
    },
    "zh": {
        "Title": "<h1 style='text-align: center;'>音訊標注為訓練用文本頁面</h1>",
        "LanguageModel": "選擇音訊的語言，將影響訓練後模型類型",
        "WhisperSize": "選擇Whisper模型大小，越大精準度越高",
        "DenoiseAudioDir": "來源音訊目錄(去噪音訊目錄)",
        "TargetSampleRate": "目標取樣率",
        "MergeCount": "針對語速比較快的配音員，可以使用Merge Count將多個句子合併成一個句子，避免訓練語音太短的問題",
        "Submit": "提交",
        "Output": "輸出結果",
        "TranscribeComplete": "標注完成",
    },
    "ja": {
        "Title": "<h1 style='text-align: center;'>トレーニングテキスト用のオーディオ転写ページ</h1>",
        "LanguageModel": "音声の言語を選択し、トレーニング後のモデルタイプに影響します",
        "WhisperSize": "Whisperモデルサイズを選択します。サイズが大きいほど精度が高くなります",
        "DenoiseAudioDir": "ソースオーディオディレクトリ（ノイズ除去オーディオディレクトリ）",
        "TargetSampleRate": "ターゲットサンプリングレート",
        "MergeCount": "話す速度が速い声優の場合、Merge Countを使用して複数の文を1つに結合し、トレーニング音声が短すぎる問題を回避できます",
        "Submit": "提出",
        "Output": "出力",
        "TranscribeComplete": "転写完了",
    },
}

_page_prefix = "transcribe_audio"
_translations = add_prefix_to_translations(_translations, _page_prefix)

def submit(denoise_audio_dir, language_model, whisper_size, target_sr, merge_count):
    with set_page_prefix(_page_prefix):
        long_audio.transcribe_audio(denoise_audio_dir, language_model, whisper_size, target_sr, merge_count)
        return gettext("transcribe complete")

def create_transcribe_audio_interface(lang):
    with set_page_prefix(_page_prefix):
        with gr.Blocks() as transcribe_audio_blocks:
            title = gr.Markdown(gettext("Title"))
            language_model = gr.Dropdown(
                label=gettext("LanguageModel"),
                choices=[("Chinese", "C"), ("Chinese+Japanese", "CJ"), ("Chinese+Japanese+English", "CJE")],
                value="C"
            )
            whisper_size = gr.Dropdown(
                label=gettext("WhisperSize"),
                choices=[
                    ("tiny (39 M)", "tiny"),
                    ("base (74 M)", "base"),
                    ("small (244 M)", "small"),
                    ("medium (769 M)", "medium"),
                    ("large (1550 M)", "large"),
                    ("large-v2 (1550 M)", "large-v2")
                ],
                value="large-v2"
            )
            denoise_audio_dir = gr.Textbox(label=gettext("DenoiseAudioDir"), value="training_data/denoised_audio/")
            target_sr = gr.Textbox(label=gettext("TargetSampleRate"), value="16000")
            merge_count = gr.Number(label=gettext("MergeCount"), value=1)
            submit_button = gr.Button(gettext("Submit"), variant="primary")
            output = gr.Textbox(label=gettext("Output"))

            submit_button.click(submit, inputs=[denoise_audio_dir, language_model, whisper_size, target_sr, merge_count], outputs=[output])

    translate_blocks(block=transcribe_audio_blocks, translation=_translations, lang=lang)