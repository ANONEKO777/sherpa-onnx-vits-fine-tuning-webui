import os
import json
import argparse
import torchaudio


def main():
    with open("./configs/finetune_speaker.json", 'r', encoding='utf-8') as f:
        hps = json.load(f)
    target_sr = hps['data']['sampling_rate']
    dir_path = "sampled_audio4ft"
    target_dir_path = "resampled_audio4ft"

    # [0][0]是資料夾名稱，[0][1]是資料夾底下的資料夾名稱，[0][2]是資料夾底下的檔案名稱
    dir_list = list(os.walk(dir_path))[0][1]
    for directory in dir_list:
        filelist = list(os.walk(os.path.join(dir_path, directory)))[0][2]
        for wavfile in filelist:
            file_path = os.path.join(dir_path, directory, wavfile)
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"The file '{file_path}' does not exist.")
            
            try:
                wav, sr = torchaudio.load(file_path, frame_offset=0, num_frames=-1, normalize=True, channels_first=True)
                print(f"Resampling {file_path}, sr={sr}")
                if sr != target_sr:
                    wav = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)(wav)
                target_file_path = os.path.join(target_dir_path, directory, wavfile)
                os.makedirs(os.path.dirname(target_file_path), exist_ok=True)
                torchaudio.save(target_file_path, wav, target_sr, channels_first=True)
            except Exception as e:
                print(f"Error processing file {file_path}: {e}")

if __name__ == "__main__":
    main()