#!/usr/bin/env python3
# author: ANONEKO
# License: MIT

"""
這個腳本用來轉換由Plachtaa開發的VITS-fast-fine-tuning模型。請注意python使用的是3.8版本，其他版本可能會有問題。

This script is used to convert the VITS-fast-fine-tuning model developed by Plachtaa. Please note that python uses version 3.8, other versions may have problems.

# 使用方法
Usage:

1. 下載預訓練模型 Download pre-trained models
Linux
```bash
wget https://huggingface.co/spaces/zomehwh/vits-uma-genshin-honkai/resolve/main/model/G_953000.pth -P models/vits-uma-genshin-honkai
wget https://huggingface.co/spaces/zomehwh/vits-uma-genshin-honkai/resolve/main/model/config.json -P models/vits-uma-genshin-honkai
```
Windows
```bash
Invoke-WebRequest -Uri https://huggingface.co/spaces/zomehwh/vits-uma-genshin-honkai/resolve/main/model/G_953000.pth -OutFile models/vits-uma-genshin-honkai/G_953000.pth
Invoke-WebRequest -Uri https://huggingface.co/spaces/zomehwh/vits-uma-genshin-honkai/resolve/main/model/config.json -OutFile models/vits-uma-genshin-honkai/config.json
```

2. 建立虛擬環境 Create a virtual environment
```bash
python3 -m venv venv
```

3. 啟動虛擬環境 Activate the virtual environment
Linux
```bash
source venv/bin/activate
```
Windows
```bash
venv\Scripts\activate
```

4. 安裝依賴 Install dependencies
```bash
pip install -r requirements.txt
```

5. 執行腳本 Run this file

```bash
python export-vits-fast-fine-tuning-onnx.py --config ./models/vits-uma-genshin-honkai/config.json --checkpoint ./models/vits-uma-genshin-honkai/G_953000.pth
```

"""
import sys

sys.path.insert(0, "./VITS-fast-fine-tuning")  # noqa

import argparse
import os
from pathlib import Path
from typing import Dict, Any, List

from pypinyin import load_phrases_dict, phrases_dict, pinyin_dict

new_phrases = {
    "行長": [],
    "還我": [],
}

load_phrases_dict(new_phrases)

import commons
import onnx
import torch
import utils
import shutil
from models import SynthesizerTrn
from onnxruntime.quantization import QuantType, quantize_dynamic
from text import _clean_text, text_to_sequence
from text.symbols import symbols
from text.symbols import _punctuation

_ouput_dir = Path("onnx-output")

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default=_ouput_dir,
    )

    parser.add_argument(
        "--comment",
        type=str,
        default="",
    )

    parser.add_argument(
        "--language",
        type=str,
        default="Chinese",
    )

    parser.add_argument(
        "--model_name",
        type=str,
        default="vits",
    )

    return parser.parse_args()


class OnnxModel(torch.nn.Module):
    def __init__(self, model: SynthesizerTrn):
        super().__init__()
        self.model = model

    def forward(
        self,
        x,
        x_lengths,
        noise_scale=1,
        length_scale=1,
        noise_scale_w=1.0,
        sid=0,
        max_len=None,
    ):
        return self.model.infer(
            x=x,
            x_lengths=x_lengths,
            sid=sid,
            noise_scale=noise_scale,
            length_scale=length_scale,
            noise_scale_w=noise_scale_w,
            max_len=max_len,
        )[0]


def get_text(text, hps, is_symbol):
    text_norm = text_to_sequence(text, hps.symbols, [] if is_symbol else hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = torch.LongTensor(text_norm)
    return text_norm

def get_phones(word, text_cleaners) -> List[str]:
    text = f"[ZH]{word}[ZH]"
    phones: str = _clean_text(text, text_cleaners)
    return list(phones)[:-1]

def check_args(args):
    assert Path(args.config).is_file(), args.config
    assert Path(args.checkpoint).is_file(), args.checkpoint

def add_meta_data(filename: str, meta_data: Dict[str, Any]):
    """Add meta data to an ONNX model. It is changed in-place.

    Args:
      filename:
        Filename of the ONNX model to be changed.
      meta_data:
        Key-value pairs.
    """
    model = onnx.load(filename)
    for key, value in meta_data.items():
        meta = model.metadata_props.add()
        meta.key = key
        meta.value = str(value)

    onnx.save(model, filename)


def generate_tokens(hps, output_path):
    with open(os.path.join(output_path, "tokens.txt"), "w", encoding="utf-8") as f:
        for i, s in enumerate(hps.symbols):
            f.write(f"{s} {i}\n")
    print("Generated tokens.txt")

def generate_lexicon(hps, output_path):
    words = list()

    phrases = phrases_dict.phrases_dict
    word_dict = pinyin_dict.pinyin_dict
    for key in word_dict:
        if not (0x4E00 <= key <= 0x9FFF):
            continue
        w = chr(key)
        words.append(w)

    for key in phrases:
        words.append(key)

    for key in new_phrases:
        words.append(key)

    symbol_to_id = {s: i for i, s in enumerate(hps.symbols)}
    word2phone = []
    for w in words:
        phones = get_phones(w, hps.data.text_cleaners)
        oov = False
        for p in phones:
            if p not in symbol_to_id:
                oov = True
                break
        if oov:
            print(f"Skip {w}")
            continue

        word2phone.append([w, " ".join(phones)])

    with open(os.path.join(output_path, "lexicon.txt"), "w", encoding="utf-8") as f:
        for w, phones in word2phone:
            f.write(f"{w} {phones}\n")
    print("Generated lexicon.txt")

@torch.no_grad()
def export_onnx_model(hps, args):
    """
    將vits-fast-fine-tuning模型轉換為onnx格式
    :param hps: config.json中的參數
    :param args: 命令行參數
    """
    # 輸出目錄包含模型名稱
    # Output directory contains model name
    output_path = os.path.join(args.output_dir, args.model_name)
    # 創建目錄
    # Create directory
    Path(output_path).mkdir(parents=True, exist_ok=True)

    generate_tokens(hps, output_path)
    generate_lexicon(hps, output_path)

    net_g = SynthesizerTrn(
        len(hps.symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        **hps.model)
    _ = net_g.eval()

    _ = utils.load_checkpoint(args.checkpoint, net_g, None)

    x = get_text("Liliana is the most beautiful assistant", hps, False)
    x = x.unsqueeze(0)

    x_length = torch.tensor([x.shape[1]], dtype=torch.int64)
    noise_scale = torch.tensor([1], dtype=torch.float32)
    length_scale = torch.tensor([1], dtype=torch.float32)
    noise_scale_w = torch.tensor([1], dtype=torch.float32)
    sid = torch.tensor([0], dtype=torch.int64)

    model = OnnxModel(net_g)

    opset_version = 13

    filename = os.path.join(output_path, "model.onnx")

    torch.onnx.export(
        model,
        (x, x_length, noise_scale, length_scale, noise_scale_w, sid),
        filename,
        opset_version=opset_version,
        input_names=[
            "x",
            "x_length",
            "noise_scale",
            "length_scale",
            "noise_scale_w",
            "sid",
        ],
        output_names=["y"],
        dynamic_axes={
            "x": {0: "N", 1: "L"},  # n_audio is also known as batch_size
            "x_length": {0: "N"},
            "y": {0: "N", 2: "L"},
        },
    )
    meta_data = {
        "model_type": "vits",
        "comment": args.comment,
        "language": args.language,
        "jieba": 1,
        "add_blank": int(hps.data.add_blank),
        "n_speakers": int(hps.data.n_speakers),
        "sample_rate": hps.data.sampling_rate,
        "punctuation": " ".join(list(_punctuation)),
    }
    print("meta_data", meta_data)
    add_meta_data(filename=filename, meta_data=meta_data)

    print("Generate int8 quantization models")

    filename_int8 = os.path.join(output_path, "model.int8.onnx")
    quantize_dynamic(
        model_input=filename,
        model_output=filename_int8,
        weight_type=QuantType.QUInt8,
    )

    # 將source-for-output的檔案和資料夾全都複製到output_dir
    # Copy all files and folders from source-for-output to output_dir
    # 使用 shutil.copytree 複製整個目錄
    source_dir = Path("source-for-output")
    try:
        shutil.copytree(source_dir, output_path, dirs_exist_ok=True)
        print(f"Successfully copied {source_dir} to {output_path}")
    except Exception as e:
        print(f"Error copying directory: {e}")


    print(f"Saved to {filename} and {filename_int8}")

def main():
    args = get_args()
    check_args(args)
    hps = utils.get_hparams_from_file(args.config)
    export_onnx_model(hps, args)
    
if __name__ == "__main__":
    main()
