#!/usr/bin/env python3
# author: ANONEKO
# License: MIT

"""
這個腳本用來推理ONNX模型，將文字轉換為語音。

This script is used to infer ONNX models, converting text to speech.

# 使用方法
Usage:

執行腳本 Run this file

```bash
python onnx-inference.py --checkpoint ./onnx-output/vits-uma-genshin-honkai/model.onnx --lexicon ./onnx-output/vits-uma-genshin-honkai/lexicon.txt --tokens ./onnx-output/vits-uma-genshin-honkai/tokens.txt
```
"""

from typing import Dict, List

import argparse
from pathlib import Path
import jieba
import onnxruntime
import soundfile
import torch

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--text",
        type=str,
        default="大家早安，歡迎來到文字轉語音的世界！",
    )

    parser.add_argument(
        "--sid",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--lexicon",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--tokens",
        type=str,
        required=True,
    )

    return parser.parse_args()


def check_args(args):
    assert Path(args.checkpoint).is_file(), args.checkpoint
    assert Path(args.lexicon).is_file(), args.lexicon
    assert Path(args.tokens).is_file(), args.tokens

def display(sess):
    for i in sess.get_inputs():
        print(i)

    print("-" * 10)
    for o in sess.get_outputs():
        print(o)


class OnnxModel:
    def __init__(
        self,
        model: str,
    ):
        session_opts = onnxruntime.SessionOptions()
        session_opts.inter_op_num_threads = 1
        session_opts.intra_op_num_threads = 4

        self.session_opts = session_opts

        self.model = onnxruntime.InferenceSession(
            model,
            sess_options=self.session_opts,
        )
        display(self.model)

        meta = self.model.get_modelmeta().custom_metadata_map
        self.add_blank = int(meta["add_blank"])
        self.sample_rate = int(meta["sample_rate"])
        self.punctuation = meta["punctuation"].split()
        print(meta)

    def __call__(
        self,
        x: torch.Tensor,
        sid: int,
    ) -> torch.Tensor:
        """
        Args:
          x:
            A int64 tensor of shape (L,)
        """
        x = x.unsqueeze(0)
        x_length = torch.tensor([x.shape[1]], dtype=torch.int64)
        noise_scale = torch.tensor([1], dtype=torch.float32)
        length_scale = torch.tensor([1], dtype=torch.float32)
        noise_scale_w = torch.tensor([1], dtype=torch.float32)
        sid = torch.tensor([sid], dtype=torch.int64)

        y = self.model.run(
            [
                self.model.get_outputs()[0].name,
            ],
            {
                self.model.get_inputs()[0].name: x.numpy(),
                self.model.get_inputs()[1].name: x_length.numpy(),
                self.model.get_inputs()[2].name: noise_scale.numpy(),
                self.model.get_inputs()[3].name: length_scale.numpy(),
                self.model.get_inputs()[4].name: noise_scale_w.numpy(),
                self.model.get_inputs()[5].name: sid.numpy(),
            },
        )[0]
        return torch.from_numpy(y).squeeze()


def read_lexicon(path) -> Dict[str, List[str]]:
    ans = dict()
    with open(path, encoding="utf-8") as f:
        for line in f:
            w_p = line.split()
            w = w_p[0]
            p = w_p[1:]
            ans[w] = p
    return ans


def read_tokens(path) -> Dict[str, int]:
    ans = dict()
    with open(path, encoding="utf-8") as f:
        for line in f:
            t_i = line.strip().split()
            if len(t_i) == 1:
                token = " "
                idx = t_i[0]
            else:
                assert len(t_i) == 2, (t_i, line)
                token = t_i[0]
                idx = t_i[1]
            ans[token] = int(idx)
    return ans


def convert_lexicon(lexicon, tokens):
    for w in lexicon:
        phones = lexicon[w]
        try:
            p = [tokens[i] for i in phones]
            lexicon[w] = p
        except Exception:
            #  print("skip", w)
            continue


"""
skip rapprochement
skip croissants
skip aix-en-provence
skip provence
skip croissant
skip denouement
skip hola
skip blanc
"""


def get_text(text, lexicon, tokens, punctuation):
    # 使用 jieba 進行分詞
    text = jieba.lcut(text.lower())
    ans = []
    for i in range(len(text)):
        w = text[i]
        punct = None

        if w and w[0] in punctuation:
            ans.append(tokens[w[0]])
            w = w[1:]

        if w and w[-1] in punctuation:
            # 使用 dict.get 方法來避免 KeyError
            punct = tokens.get(w[-1], None)
            if punct is None:
                punct = tokens[" "]
            w = w[:-1]

        if w in lexicon:
            ans.extend(lexicon[w])
            if punct:
                ans.append(punct)

            if i != len(text) - 1:
                ans.append(tokens[" "])
            continue

        # 如果 w 不在 lexicon 中，將其分成單個字再重新查找
        for char in w:
            if char in lexicon:
                ans.extend(lexicon[char])
            else:
                print("ignore", char)
            ans.append(tokens[" "])  # 添加空格作為分隔符

        if punct:
            ans.append(punct)

    print("ans = ", ans)
    return ans

def generate(model, text, lexicon, tokens, sid):
    x = get_text(
        text,
        lexicon,
        tokens,
        model.punctuation,
    )
    if model.add_blank:
        x2 = [0] * (2 * len(x) + 1)
        x2[1::2] = x
        x = x2

    x = torch.tensor(x, dtype=torch.int64)

    y = model(x, sid=sid)

    return y

def infer(model, text, lexicon, tokens, sid):
    """
    進行文字轉語音的推理
    :param model: ONNX模型
    :param text: 輸出語音的文字
    :param lexicon: 字典
    :param tokens: 字典
    :param sid: 語音樣本的ID，也就是說話人的ID
    :return: 輸出語音的波形和採樣率
    Args:
        text: str
        lexicon: Dict[str, List[str]]
        tokens: Dict[str, int]
        sid: int
    """
    y = generate(model, text, lexicon, tokens, sid)
    return model.sample_rate, y.numpy()

def infer_by_args(args):
    model = OnnxModel(args.checkpoint)

    lexicon = read_lexicon(args.lexicon)
    tokens = read_tokens(args.tokens)
    convert_lexicon(lexicon, tokens)

    return infer(model, args.text, lexicon, tokens, args.sid)

def main():
    args = get_args()
    check_args(args)
    # 初始化模型
    model = OnnxModel(args.checkpoint)

    lexicon = read_lexicon(args.lexicon)
    tokens = read_tokens(args.tokens)
    convert_lexicon(lexicon, tokens)

    sample_rate, audio = infer(model, args.text, lexicon, tokens, args.sid)
    soundfile.write("test.wav", audio, sample_rate)


if __name__ == "__main__":
    main()
