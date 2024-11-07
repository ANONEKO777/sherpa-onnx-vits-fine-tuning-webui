<div align="center">
<h1>Sherpa-Onnx-VITS-Fine-Tuning-WebUI</h1>

集合VITS推理、訓練與輸出Sherpa-Onnx格式的一頁式WebUI

[![Static Badge](https://img.shields.io/badge/made_with-%F0%9F%92%96-red?style=for-the-badge&labelColor=orange)](https://github.com/anoneko777/sherpa-onnx-vits-fine-tuning-webui)
[![Open In Colab](https://img.shields.io/badge/Colab-F9AB00?style=for-the-badge&logo=googlecolab&color=525252)](https://github.com/anoneko777/sherpa-onnx-vits-fine-tuning-webui)
[![License](https://img.shields.io/badge/LICENSE-MIT-green.svg?style=for-the-badge)](https://github.com/anoneko777/sherpa-onnx-vits-fine-tuning-webui/blob/main/LICENSE)
[![Huggingface](https://img.shields.io/badge/🤗%20-online%20demo-yellow.svg?style=for-the-badge)](https://github.com/anoneko777/sherpa-onnx-vits-fine-tuning-webui)

[**English**](../README.md) | **正體中文** | [**日本語**](README-ja.md)

</div>

---

# 介紹
本專案提供一頁式簡易操作的WebUI，可使用自己的音源檔案訓練VITS模型，並測試推理效果，再輸出可供Sherpa-Onnx框架使用的Onnx模型格式。其技術基於Plachtaa開發的[VITS-fast-fine-tuning](https://github.com/Plachtaa/VITS-fast-fine-tuning)與Kaldi團隊開發的[Sherpa-Onnx](https://github.com/k2-fsa/sherpa-onnx)。

本專案的開發目標是為了更方便訓練自己的語音模型，並讓語音模型能在Android、iOS等移動端平台使用。

> [!IMPORTANT]  
> 請注意，本專案目前仍處於開發階段，使用上可能會出現錯誤或bug，請見諒。歡迎和我聯繫或提供issue與解決方案。

## 為什麼使用VITS?
VITS為端到端文字轉語音 (TTS) 模型，提供自然且流露情緒的語音生成功能，並且在CPU推理下仍有出色的執行效率。本專案立項時已有如Bert-VITS2、GPT-SoVITS、FishSpeech等效果更優異的TTS模型，但通常需要使用GPU推理且占用更多的vram。VITS-fast-fine-tuning提供英文、中文、日文的語音訓練。

## 為什麼使用Sherpa-Onnx?
Sherpa-Onnx框架使用新一代 Kaldi 和 onnxruntime 進行語音轉文字、文字轉語音、說話者辨識和 VAD，無需連接網路。支援嵌入式系統、Android, iOS, Raspberry Pi, RISC-V, x86_64 servers, websocket server/client, C/C++, Python, Kotlin, C#, Go, NodeJS, Java, Swift, Dart, JavaScript, Flutter, Object Pascal, Lazarus, Rust。因此將VITS訓練後的模型格式轉換為Sherpa-Onnx，能更有效在多平台上使用。

# Todo List

- [x] **提供轉換用的腳本。**
- [x] **提供可供操作的WebUI。**
  - [x] Step-by-Step的訓練操作介面。
  - [x] 支援從Youtube上批量下載語音。
  - [ ] 支援短語音、長語音、影片訓練。
  - [ ] 支援長語音切分成短語音功能。
  - [ ] 支援語音降躁，支援whisper語音轉文字辨識，標註文件。
  - [ ] 訓練後可直接用模型推理。
  - [x] 支援Pytorch模型轉換為Sherpa-Onnx模型。
  - [x] Sherpa-Onnx模型推理。
- [ ] 提供一鍵安裝自動執行所有安裝指令。
- [ ] 提供Huggingface Online Demo。
- [ ] 提供Colab腳本。

# 安裝與使用
> [!TIP]  
> 本專案使用的python是3.8版本，其他版本可能會有問題。

## 將預訓練模型轉換為Sherpa-Onnx格式
### 1. 下載預訓練模型
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

### 2. 建立虛擬環境
```bash
python3 -m venv venv
```

### 3. 啟動虛擬環境
Linux
```bash
source venv/bin/activate
```
Windows
```bash
venv\Scripts\activate
```

### 4. 安裝依賴
```bash
pip install -r requirements.txt
python vits_fast_fine_tuning/monotonic_align/setup.py build_ext
```

### 5. 執行腳本

```bash
python export_vits_fast_fine_tuning_onnx.py --config ./models/vits-uma-genshin-honkai/config.json --checkpoint ./models/vits-uma-genshin-honkai/G_953000.pth
```
命令列的可用參數說明
 - --config 必要 vits訓練完成後產生的config檔。
 - --checkpoint 必要 vits訓練完成後產生的pytorch模型。
 - --output_dir 可選 輸出onnx模型的資料夾，預設是onnx-output。
 - --comment 可選 onnx模型內的註解資訊。
 - --language 可選 onnx模型內的語言資訊。
 - --model_name 可選 模型名稱，主要使用在輸出資料夾內建立子資料夾時的名稱。

### 6. 輸出結果
輸出成功後在資料夾onnx-output內應該能找到模型名稱的資料夾，資料夾底下應該會有以下檔案與結構，這些檔案都是sherpa-onnx會用到的。model.onnx是原始模型，model.int8.onnx是量化後的模型，兩者擇一即可。model-opt.onnx是優化後的模型，目前尚不確定sherpa-onnx是否支援。
```bash
│  date.fst
│  lexicon.txt
│  model-opt.onnx
│  model.int8.onnx
│  model.onnx
│  new_heteronym.fst
│  number.fst
│  phone.fst
│  tokens.txt
│
└─dict
    │  hmm_model.utf8
    │  idf.utf8
    │  jieba.dict.utf8
    │  README.md
    │  stop_words.utf8
    │  user.dict.utf8
    │
    └─pos_dict
            char_state_tab.utf8
            prob_emit.utf8
            prob_start.utf8
            prob_trans.utf8
```

### 7. 推理
```bash
python onnx_inference.py --checkpoint ./onnx-output/vits-uma-genshin-honkai/model.onnx --lexicon ./onnx-output/vits-uma-genshin-honkai/lexicon.txt --tokens ./onnx-output/vits-uma-genshin-honkai/tokens.txt
```
命令列的可用參數說明
 - --checkpoint 必要 onnx模型。
 - --lexicon 必要 模型使用的音素表。
 - --token 必要 模型使用的符號表。
 - --text 可選 用來文字轉語音的內容。

## 啟動WebUI
> [!NOTE]  
> 功能尚在開發中。

```bash
python webui.py
```

## 訓練指令
> [!NOTE]  
> 功能尚在開發中。
```bash
# 已改用yt-dlp，原本該專案使用的youtube-dl已經無法下載影片
python scripts/download_video.py
# 還有bug
python scripts/video2audio.py
python scripts/denoise_audio.py
# 注意whisper必須要有ffmpeg
python scripts/long_audio_transcribe.py --languages "C" --whisper_size large-v2
```

請注意，如果訓練的時候發現無法找到ffmpeg，如以下的錯誤訊息，則需要安裝ffmpeg的library。
```bash
DEBUG:torio._extension.utils:Loading FFmpeg
DEBUG:torio._extension.utils:Failed to load FFmpeg extension.
```

根據torch的[官方說明](https://pytorch.org/audio/2.3.0/installation.html)，透過下述的命名法則來尋找library，如果有安裝上的問題可先找找是否能找到相關的檔案。
> When searching for FFmpeg installation, TorchAudio looks for library files which have names with version numbers. That is, libavutil.so.<VERSION> for Linux, libavutil.<VERSION>.dylib for macOS, and avutil-<VERSION>.dll for Windows. Many public pre-built binaries follow this naming scheme, but some distributions have un-versioned file names. If you are having difficulties detecting FFmpeg, double check that the library files you installed follow this naming scheme, (and then make sure that they are in one of the directories listed in library search path.)

> 當搜尋 FFmpeg 安裝時，TorchAudio 會尋找名稱帶有版本號的庫檔案。即， libavutil.so.<VERSION> （適用於 Linux）、 libavutil.<VERSION>.dylib （適用於 macOS）和avutil-<VERSION>.dll （適用於 Windows）。許多公共預先建置的二進位檔案都遵循此命名方案，但某些發行版具有未版本化的檔案名稱。如果您在偵測 FFmpeg 時遇到困難，請仔細檢查您安裝的程式庫檔案是否遵循此命名方案（然後確保它們位於庫搜尋路徑中列出的目錄之一中。）

Windows版本安裝ffmpeg可以透過以下的github找到編譯好的library。
https://github.com/BtbN/FFmpeg-Builds/releases

要特別注意要找的是有包含library的版本，且因為本專案使用的torchaudio需要用版本6、5、4的ffmpeg，建議可從下列網址進行下載。
https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-n6.1-latest-win64-lgpl-shared-6.1.zip

