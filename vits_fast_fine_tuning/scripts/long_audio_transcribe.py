from moviepy.editor import AudioFileClip
import whisper
import os
import json
import torchaudio
import librosa
import torch
import argparse

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--languages",
        type=str,
        default="CJE",
    )

    parser.add_argument(
        "--whisper_size",
        type=str,
        default="medium"
    )

    parser.add_argument(
        "--denoise_audio_dir",
        type=str,
        default="./training_data/denoised_audio/",
    )

    parser.add_argument(
        "--merge_count",
        type=int,
        default=2,
        help="Number of sentences to merge in segment_audio_merged"
    )

    return parser.parse_args()

def segment_audio_directly(result, lang2token, lang, wav, sr, character_name, code, target_sr):
    speaker_annos = []
    for i, seg in enumerate(result['segments']):
        start_time = seg['start']
        end_time = seg['end']
        text = seg['text']
        text = lang2token[lang] + text.replace("\n", "") + lang2token[lang]
        text = text + "\n"
        wav_seg = wav[:, int(start_time*sr):int(end_time*sr)]
        wav_seg_name = f"{character_name}_{code}_{i}.wav"
        savepth = "training_data/segmented_character_voice/" + character_name + "/" + wav_seg_name
        speaker_annos.append(savepth + "|" + character_name + "|" + text)
        print(f"Transcribed segment: {speaker_annos[-1]}")
        torchaudio.save(savepth, wav_seg, target_sr, channels_first=True)
    return speaker_annos

def segment_audio_merged(result, lang2token, lang, wav, sr, character_name, code, target_sr, merge_count=2):
    speaker_annos = []
    merged_segments = []
    i = 0
    while i < len(result['segments']):
        start_time = result['segments'][i]['start']
        end_time = result['segments'][i]['end']
        text = result['segments'][i]['text']

        for _ in range(merge_count - 1):
            if i + 1 < len(result['segments']):
                end_time = result['segments'][i + 1]['end']
                text += result['segments'][i + 1]['text']
                i += 1
            else:
                break

        merged_segments.append({
            'start': start_time,
            'end': end_time,
            'text': text
        })
        i += 1

    for i, seg in enumerate(merged_segments):
        start_time = seg['start']
        end_time = seg['end']
        text = seg['text']
        text = lang2token[lang] + text.replace("\n", "") + lang2token[lang]
        text = text + "\n"
        wav_seg = wav[:, int(start_time*sr):int(end_time*sr)]
        wav_seg_name = f"{character_name}_{code}_{i}.wav"
        savepth = "training_data/segmented_character_voice/" + character_name + "/" + wav_seg_name
        speaker_annos.append(savepth + "|" + character_name + "|" + text)
        print(f"Transcribed segment: {speaker_annos[-1]}")
        torchaudio.save(savepth, wav_seg, target_sr, channels_first=True)
    return speaker_annos

def transcribe_audio(denoise_audio_dir, languages, whisper_size, merge_count):
    if languages == "CJE":
        lang2token = {
            'zh': "[ZH]",
            'ja': "[JA]",
            "en": "[EN]",
        }
    elif languages == "CJ":
        lang2token = {
            'zh': "[ZH]",
            'ja': "[JA]",
        }
    elif languages == "C":
        lang2token = {
            'zh': "[ZH]",
        }

    if not torch.cuda.is_available():
        print("Please enable GPU in order to run Whisper!")

    with open("training_data/configs/finetune_speaker.json", 'r', encoding='utf-8') as f:
        hps = json.load(f)
    target_sr = hps['data']['sampling_rate']

    filelist = list(os.walk(denoise_audio_dir))[0][2]

    model = whisper.load_model(whisper_size)
    speaker_annos = []
    for file in filelist:
        if not file.endswith(".wav"):
            continue
        options = dict(beam_size=5, best_of=5)
        transcribe_options = dict(task="transcribe", **options)
        print(f"transcribing {os.path.join(denoise_audio_dir, file)}\n")
        result = model.transcribe(os.path.join(denoise_audio_dir, file), word_timestamps=True, **transcribe_options)
        lang = result['language']
        if result['language'] not in list(lang2token.keys()):
            print(f"{lang} not supported, ignoring...\n")
            continue
        # segment audio based on segment results
        character_name = file.rstrip(".wav").split("_")[0]
        code = file.rstrip(".wav").split("_")[1]
        if not os.path.exists("training_data/segmented_character_voice/" + character_name):
            os.makedirs("training_data/segmented_character_voice/" + character_name)
        wav, sr = torchaudio.load(os.path.join(denoise_audio_dir, file), frame_offset=0, num_frames=-1, normalize=True,
                                  channels_first=True)
        if merge_count >= 2:
            speaker_annos += segment_audio_merged(result, lang2token, lang, wav, sr, character_name, code, target_sr, merge_count)
        else:
            speaker_annos += segment_audio_directly(result, lang2token, lang, wav, sr, character_name, code, target_sr)

    if len(speaker_annos) == 0:
        print("Warning: no long audios & videos found, this IS expected if you have only uploaded short audios")
        print("this IS NOT expected if you have uploaded any long audios, videos or video links. Please check your file structure or make sure your audio/video language is supported.")

    with open("training_data/long_character_anno.txt", 'w', encoding='utf-8') as f:
        for line in speaker_annos:
            f.write(line)

def main():
    args = get_args()
    transcribe_audio(args.denoise_audio_dir, args.languages, args.whisper_size, args.merge_count)

if __name__ == "__main__":
    main()