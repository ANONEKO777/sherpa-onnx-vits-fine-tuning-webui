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

- [ ] **Features:**
  - [x] 提供轉換用的腳本。
  - [ ] 提供可供操作的WebUI。

# 安裝與使用
> [!TIP]  
> 本專案使用的python是3.8版本，其他版本可能會有問題。

## 將預訓練模型轉換為Sherpa-Onnx格式
### 1. 下載預訓練模型
Linux
```bash
wget https://huggingface.co/spaces/zomehwh/vits-uma-genshin-honkai/resolve/main/model/G_953000.pth -P model/vits-uma-genshin-honkai
wget https://huggingface.co/spaces/zomehwh/vits-uma-genshin-honkai/resolve/main/model/config.json -P model/vits-uma-genshin-honkai
```
Windows
```bash
Invoke-WebRequest -Uri https://huggingface.co/spaces/zomehwh/vits-uma-genshin-honkai/resolve/main/model/G_953000.pth -OutFile model/vits-uma-genshin-honkai/G_953000.pth
Invoke-WebRequest -Uri https://huggingface.co/spaces/zomehwh/vits-uma-genshin-honkai/resolve/main/model/config.json -OutFile model/vits-uma-genshin-honkai/config.json
```

### 2. 建立虛擬環境
```bash
python3 -m venv fast_vits_env
```

### 3. 啟動虛擬環境
Linux
```bash
source fast_vits_env/bin/activate
```
Windows
```bash
fast_vits_env\Scripts\activate
```

### 4. 安裝依賴
```bash
pip install -r requirements.txt
```

### 5. 執行腳本

```bash
./export-vits-fast-fine-tuning-onnx.py --config ./model/vits-uma-genshin-honkai/config.json --checkpoint ./model/vits-uma-genshin-honkai/G_953000.pth
```

## 啟動WebUI
> [!NOTE]  
> 功能尚在開發中。

