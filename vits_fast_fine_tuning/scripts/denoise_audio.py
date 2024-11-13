#!/usr/bin/env python3
# author: ANONEKO
# License: MIT
import os
import argparse
import json

import torchaudio

from . import demucs_api

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--raw_audio_dir",
        type=str,
        default="./training_data/raw_audio/",
    )

    parser.add_argument(
        "--denoise_audio_dir",
        type=str,
        default="./training_data/denoised_audio/",
    )

    return parser.parse_args()

def check_args(args):
    assert os.path.isdir(args.raw_audio_dir), args.raw_audio_dir

def denoise_audio(raw_audio_dir, denoise_audio_dir):
    # 檢查raw_audio_dir是否存在，不存在跳錯誤訊息
    if not os.path.isdir(raw_audio_dir):
        print(f"raw_audio_dir: {raw_audio_dir} not found!")
        return
    
    filelist = list(os.walk(raw_audio_dir))[0][2]

    if not os.path.exists(denoise_audio_dir):
        os.makedirs(denoise_audio_dir)

    # Get the target sampling rate
    with open("training_data/configs/finetune_speaker.json", 'r', encoding='utf-8') as f:
        hps = json.load(f)
    target_sr = hps['data']['sampling_rate']

    separator = demucs_api.Separator()

    for file in filelist:
        if file.endswith(".wav"):
            origin, separated = separator.separate_audio_file(os.path.join(raw_audio_dir, file))
            for stem, source in separated.items():
                # merge two channels into one
                source = source.mean(dim=0).unsqueeze(0)
                # 重新採樣
                source = torchaudio.transforms.Resample(orig_freq=separator.model.samplerate, new_freq=target_sr)(source)
                torchaudio.save(os.path.join(denoise_audio_dir, file), source, target_sr, channels_first=True)

def main():
    args = get_args()
    check_args(args)
    denoise_audio(args.raw_audio_dir, args.denoise_audio_dir)

if __name__ == "__main__":
    main()