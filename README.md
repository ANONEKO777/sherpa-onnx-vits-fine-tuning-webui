<div align="center">
<h1>Sherpa-Onnx-VITS-Fine-Tuning-WebUI</h1>

A one-page WebUI integrating VITS inference, training, and output in Sherpa-Onnx format.

[![Static Badge](https://img.shields.io/badge/made_with-%F0%9F%92%96-red?style=for-the-badge&labelColor=orange)](https://github.com/anoneko777/sherpa-onnx-vits-fine-tuning-webui)
[![Open In Colab](https://img.shields.io/badge/Colab-F9AB00?style=for-the-badge&logo=googlecolab&color=525252)](https://github.com/anoneko777/sherpa-onnx-vits-fine-tuning-webui)
[![License](https://img.shields.io/badge/LICENSE-MIT-green.svg?style=for-the-badge)](https://github.com/anoneko777/sherpa-onnx-vits-fine-tuning-webui/blob/main/LICENSE)
[![Huggingface](https://img.shields.io/badge/🤗%20-online%20demo-yellow.svg?style=for-the-badge)](https://github.com/anoneko777/sherpa-onnx-vits-fine-tuning-webui)

**English** | [**正體中文**](./docs/README-zh.md) | [**日本語**](./docs/README-ja.md)

</div>

---

# Introduction
This project provides a one-page, easy-to-use WebUI that allows you to train VITS models using your own audio files, test inference results, and output models in Onnx format for use with the Sherpa-Onnx framework. The technology is based on [VITS-fast-fine-tuning](https://github.com/Plachtaa/VITS-fast-fine-tuning) developed by Plachtaa and [Sherpa-Onnx](https://github.com/k2-fsa/sherpa-onnx) developed by the Kaldi team.

The goal of this project is to make it easier to train your own speech models and to enable these models to be used on mobile platforms such as Android and iOS.

> [!IMPORTANT]  
> Please note that this project is still in the development stage, and errors or bugs may occur during use. Your understanding is appreciated. Feel free to contact me or provide issues and solutions.

## Why use VITS?
VITS is an end-to-end text-to-speech (TTS) model that provides natural and emotionally expressive speech synthesis, and it performs well even with CPU inference. At the time of this project's inception, there were more advanced TTS models like Bert-VITS2, GPT-SoVITS, and FishSpeech, but they typically require GPU inference and consume more VRAM. VITS-fast-fine-tuning offers speech training in English, Chinese, and Japanese.

## Why use Sherpa-Onnx?
The Sherpa-Onnx framework uses the next-generation Kaldi and onnxruntime for speech-to-text, text-to-speech, speaker recognition, and VAD, without requiring an internet connection. It supports embedded systems, Android, iOS, Raspberry Pi, RISC-V, x86_64 servers, websocket server/client, C/C++, Python, Kotlin, C#, Go, NodeJS, Java, Swift, Dart, JavaScript, Flutter, Object Pascal, Lazarus, and Rust. Therefore, converting the VITS-trained model format to Sherpa-Onnx allows for more effective use across multiple platforms.

# Todo List

- [ ] **Features:**
  - [x] Provide conversion scripts.
  - [ ] Provide an operational WebUI.

# Installation and Usage
> [!TIP]  
> This project uses Python version 3.8. Other versions may cause issues.

## Converting Pre-trained Models to Sherpa-Onnx Format
### 1. Download Pre-trained Models
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

### 2. Create a virtual environment
```bash
python3 -m venv fast_vits_env
```

### 3. Activate the virtual environment
Linux
```bash
source fast_vits_env/bin/activate
```
Windows
```bash
fast_vits_env\Scripts\activate
```

### 4. Install dependencies
```bash
pip install -r requirements.txt
```

### 5. Run the script

```bash
./export-vits-fast-fine-tuning-onnx.py --config ./model/vits-uma-genshin-honkai/config.json --checkpoint ./model/vits-uma-genshin-honkai/G_953000.pth
```

## Launching the WebUI
> [!NOTE]  
> This feature is still under development.

