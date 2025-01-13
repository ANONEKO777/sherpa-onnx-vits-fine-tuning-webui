import os
import argparse
import json
import sys

# from vits_fast_fine_tuning import text
from vits_fast_fine_tuning.text import _clean_text
sys.setrecursionlimit(500000)  # Fix the error message of RecursionError: maximum recursion depth exceeded while calling a Python object.  You can change the number as you want.

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--languages",
        type=str,
        default="CJE",
    )

    parser.add_argument(
        "--add_auxiliary_data",
        type=bool,
        help="Whether to add extra data as fine-tuning helper",
        default=False,
    )

    return parser.parse_args()

def modify_config(speakers, languages):
    # Determine text_cleaners based on languages
    if languages == "CJE":
        text_cleaners = ["cjke_cleaners2"]
        with open("training_data/configs/cje_finetune_speaker.json", 'r', encoding='utf-8') as f:
            hps = json.load(f)
    elif languages == "CJ":
        text_cleaners = ["zh_ja_mixture_cleaners"]
        with open("training_data/configs/cj_finetune_speaker.json", 'r', encoding='utf-8') as f:
            hps = json.load(f)
    elif languages == "C":
        text_cleaners = ["chinese_cleaners"]
        with open("training_data/configs/c_finetune_speaker.json", 'r', encoding='utf-8') as f:
            hps = json.load(f)

    # assign ids to new speakers
    speaker2id = {}
    for i, speaker in enumerate(speakers):
        speaker2id[speaker] = i
    # modify n_speakers
    hps['data']["n_speakers"] = len(speakers)
    # overwrite speaker names
    hps['speakers'] = speaker2id
    hps['train']['log_interval'] = 10
    hps['train']['eval_interval'] = 100
    hps['train']['batch_size'] = 16
    hps['data']['training_files'] = "training_data/final_annotation_train.txt"
    hps['data']['validation_files'] = "training_data/final_annotation_val.txt"
    # set text_cleaners
    hps['data']['text_cleaners'] = text_cleaners
    # save modified config
    with open("training_data/configs/modified_finetune_speaker.json", 'w', encoding='utf-8') as f:
        json.dump(hps, f, indent=2)

    return hps

def clean_anotation_file(annos, speaker2id, cleaner_names):
    cleaned_annos = []
    for i, line in enumerate(annos):
        path, speaker, txt = line.split("|")
        if len(txt) > 150:
            continue
        cleaned_text = _clean_text(txt, cleaner_names).replace("[ZH]", "")
        cleaned_text += "\n" if not cleaned_text.endswith("\n") else ""
        cleaned_annos.append(path + "|" + str(speaker2id[speaker]) + "|" + cleaned_text)
    return cleaned_annos

def save_annotation_file(final_annos, cleaned_new_annos):
    # save annotation file
    with open("training_data/final_annotation_train.txt", 'w', encoding='utf-8') as f:
        for line in final_annos:
            f.write(line)
    # save annotation file for validation
    with open("training_data/final_annotation_val.txt", 'w', encoding='utf-8') as f:
        for line in cleaned_new_annos:
            f.write(line)


def preprocess(languages, add_auxiliary_data):
    if languages == "CJE":
        langs = ["[ZH]", "[JA]", "[EN]"]
    elif languages == "CJ":
        langs = ["[ZH]", "[JA]"]
    elif languages == "C":
        langs = ["[ZH]"]
    new_annos = []
    # Source 1: transcribed short audios
    if os.path.exists("training_data/short_character_anno.txt"):
        with open("training_data/short_character_anno.txt", 'r', encoding='utf-8') as f:
            short_character_anno = f.readlines()
            new_annos += short_character_anno
    # Source 2: transcribed long audio segments
    if os.path.exists("training_data/long_character_anno.txt"):
        with open("training_data/long_character_anno.txt", 'r', encoding='utf-8') as f:
            long_character_anno = f.readlines()
            new_annos += long_character_anno

    # Get all speaker names
    speakers = []
    for line in new_annos:
        path, speaker, text = line.split("|")
        if speaker not in speakers:
            speakers.append(speaker)
    assert (len(speakers) != 0), "No audio file found. Please check your uploaded file structure."
    # Source 3 (Optional): sampled audios as extra training helpers
    if add_auxiliary_data:
        with open("training_data/sampled_audio4ft.txt", 'r', encoding='utf-8') as f:
            old_annos = f.readlines()
        # filter old_annos according to supported languages
        filtered_old_annos = []
        for line in old_annos:
            for lang in langs:
                if lang in line:
                    filtered_old_annos.append(line)
        old_annos = filtered_old_annos
        for line in old_annos:
            path, speaker, text = line.split("|")
            if speaker not in speakers:
                speakers.append(speaker)
        num_old_voices = len(old_annos)
        num_new_voices = len(new_annos)
        # STEP 1: balance number of new & old voices
        cc_duplicate = num_old_voices // num_new_voices
        if cc_duplicate == 0:
            cc_duplicate = 1

        # STEP 2: modify config file
        hps = modify_config(speakers, languages)
        speaker2id = hps['speakers']

        # STEP 3: clean annotations, replace speaker names with assigned speaker IDs
        cleaned_new_annos = clean_anotation_file(new_annos, speaker2id, hps['data']['text_cleaners'])
        cleaned_old_annos = clean_anotation_file(old_annos, speaker2id, hps['data']['text_cleaners'])

        # merge with old annotation
        final_annos = cleaned_old_annos + cc_duplicate * cleaned_new_annos
        save_annotation_file(final_annos, cleaned_new_annos)
        print("finished")
    else:
        # Do not add extra helper data
        # STEP 1: modify config file
        hps = modify_config(speakers, languages)
        speaker2id = hps['speakers']

        # STEP 2: clean annotations, replace speaker names with assigned speaker IDs
        cleaned_new_annos = clean_anotation_file(new_annos, speaker2id, hps['data']['text_cleaners'])
        final_annos = cleaned_new_annos

        save_annotation_file(final_annos, cleaned_new_annos)
        print("finished")

def main():
    args = get_args()
    preprocess(args.languages, args.add_auxiliary_data)

if __name__ == "__main__":
    main()