import os

import gradio as gr
from gradio_i18n import gettext, translate_blocks

from .commons import add_prefix_to_translations, gettext, set_page_prefix
import vits_fast_fine_tuning.finetune_speaker_v2 as finetune_speaker_v2

_translations = {
    "en": {
        "Config Path": "Config Path",
        "Model Name": "Model Name",
        "Model Name Info": "Enter the model name directly without the path. The model will be output to the models folder by default, which is the default path for the project.",
        "Max Epochs": "Max Epochs",
        "Max Epochs Info": "It is generally recommended to train for more than 100 epochs for better quality.",
        "Continue Training": "Continue Training",
        "Continue Training Info": "Make sure there are G_latest.pth and D_latest.pth in the output folder.",
        "Drop Speaker Embed": "Drop Speaker Embed",
        "Drop Speaker Embed Info": "Usually, the existing embedding data needs to be discarded when training for the first time.",
        "Train with Pretrained Model": "Train with Pretrained Model",
        "Train with Pretrained Model Info": "It is recommended to use a pre-trained model for better training quality.",
        "Number of Preserved Models": "Number of Preserved Models",
        "Run Fine-tuning": "Run Fine-tuning",
        "Output": "Output",
        "Batch Size": "Batch Size",
        "Batch Size Info": "The higher the batch size, the more VRAM it will occupy, but the training speed will be faster.",
    },
    "zh": {
        "Config Path": "設定檔路徑",
        "Model Name": "模型名稱",
        "Model Name Info": "直接輸入模型名稱，不需加上路徑，預設會輸出到models資料夾內，此為專案的預設路徑",
        "Max Epochs": "最大訓練週期",
        "Max Epochs Info": "一般建議訓練100週期以上品質會較好",
        "Continue Training": "使用前次的模型繼續訓練",
        "Continue Training Info": "請確保輸出資料夾內有G_latest.pth，D_latest.pth",
        "Drop Speaker Embed": "丟棄現存的語音角色嵌入",
        "Drop Speaker Embed Info": "通常在首次訓練時需要丟棄現存的嵌入資料",
        "Train with Pretrained Model": "使用預訓練模型訓練",
        "Train with Pretrained Model Info": "建議使用預訓練模型，訓練的品質才會比較好",
        "Number of Preserved Models": "保留模型數量",
        "Run Fine-tuning": "執行微調",
        "Output": "輸出",
        "Batch Size": "批次大小",
        "Batch Size Info": "批次越高佔用的VRAM越多，但訓練速度會加快",
    },
    "ja": {
        "Config Path": "設定ファイルのパス",
        "Model Name": "モデル名",
        "Model Name Info": "モデル名を直接入力してください。パスは不要です。モデルはデフォルトでプロジェクトのmodelsフォルダに出力されます。",
        "Max Epochs": "最大エポック数",
        "Max Epochs Info": "品質を向上させるためには、100エポック以上のトレーニングが推奨されます。",
        "Continue Training": "トレーニングを継続",
        "Continue Training Info": "出力フォルダにG_latest.pthとD_latest.pthがあることを確認してください。",
        "Drop Speaker Embed": "スピーカー埋め込みを削除",
        "Drop Speaker Embed Info": "通常、初めてトレーニングするときに既存の埋め込みデータを破棄する必要があります。",
        "Train with Pretrained Model": "事前トレーニング済みモデルでトレーニング",
        "Train with Pretrained Model Info": "品質を向上させるために、事前トレーニング済みモデルを使用することをお勧めします。",
        "Number of Preserved Models": "保存されるモデルの数",
        "Run Fine-tuning": "ファインチューニングを実行",
        "Output": "出力",
        "Batch Size": "バッチサイズ",
        "Batch Size Info": "バッチサイズが大きいほどVRAMを占有しますが、トレーニング速度が速くなります。",
    },
}

_page_prefix = "finetuning"
_translations = add_prefix_to_translations(_translations, _page_prefix)

def finetuning(config_path, model_name, max_epochs, continue_training, drop_speaker_embed, train_with_pretrained_model, preserved_models, batch_size):
    # 自動在model_name前面加上models/路徑(專案預設輸出vits模型路徑)
    model_name = os.path.join("models", f"vits-{model_name}")

    finetune_speaker_v2.run_finetuning(config_path, model_name, max_epochs, continue_training, drop_speaker_embed, train_with_pretrained_model, preserved_models, batch_size)

def create_finetuning_interface(lang):
    with set_page_prefix(_page_prefix):
        with gr.Blocks() as finetuning_blocks:
            config_path = gr.Textbox(label=gettext("Config Path"), value="training_data/configs/modified_finetune_speaker.json")
            model_name = gr.Textbox(label=gettext("Model Name"), info=gettext("Model Name Info"))
            max_epochs = gr.Number(label=gettext("Max Epochs"), value=50, info=gettext("Max Epochs Info"))
            continue_training = gr.Checkbox(label=gettext("Continue Training"), value=False, info=gettext("Continue Training Info"))
            drop_speaker_embed = gr.Checkbox(label=gettext("Drop Speaker Embed"), value=True, info=gettext("Drop Speaker Embed Info"))
            train_with_pretrained_model = gr.Checkbox(label=gettext("Train with Pretrained Model"), value=True, info=gettext("Train with Pretrained Model Info"))
            preserved_models = gr.Number(label=gettext("Number of Preserved Models"), value=4)
            batch_size = gr.Number(label=gettext("Batch Size"), value=32, info=gettext("Batch Size Info"))
            submit_button = gr.Button(value=gettext("Run Fine-tuning"), variant="primary")
            output = gr.Textbox(label=gettext("Output"))

            # 打勾continue_training時，drop_speaker_embed必須關閉；反之亦然
            continue_training.change(fn=lambda x: gr.update(value=not x), inputs=continue_training, outputs=drop_speaker_embed)

            submit_button.click(finetuning, inputs=[config_path, model_name, max_epochs, continue_training, drop_speaker_embed, train_with_pretrained_model, preserved_models, batch_size], outputs=[output])

    translate_blocks(block=finetuning_blocks, translation=_translations, lang=lang)
