<div align="center">
<h1>Sherpa-Onnx-VITS-Fine-Tuning-WebUI</h1>

A one-page WebUI integrating VITS inference, training, and output in Sherpa-Onnx format.

[![Static Badge](https://img.shields.io/badge/made_with-%F0%9F%92%96-red?style=for-the-badge&labelColor=orange)](.)
[![Open In Colab](https://img.shields.io/badge/Colab-F9AB00?style=for-the-badge&logo=googlecolab&color=525252)](.)
[![License](https://img.shields.io/badge/LICENSE-MIT-green.svg?style=for-the-badge)](./LICENSE)
[![Huggingface](https://img.shields.io/badge/🤗%20-online%20demo-yellow.svg?style=for-the-badge)](.)

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

# Feature

- [x] **Provide conversion scripts.**
- [x] **Provide an operational WebUI.**
  - [x] Step-by-step training interface.
  - [x] Support batch downloading of audio from YouTube.
  - [x] Support splitting long audio into short audio.
  - [x] Support audio denoising, whisper speech-to-text recognition, and annotation files.
  - [x] Support VITS training.
  - [ ] Support training with short audio, long audio, and video materials.
  - [x] Direct model inference after training.
  - [x] Support for converting Pytorch models to Sherpa-Onnx models.
  - [x] Sherpa-Onnx model inference.
- [ ] Provide a one-click installation script to automatically execute all installation commands.
- [ ] Provide a Huggingface Online Demo.
- [ ] Provide a Colab script.

# Installation and Usage
> [!TIP]
> This project uses Python version 3.8. Other versions may cause issues.

## Installation

### 1. Create a virtual environment
```bash
python3 -m venv venv
```

### 2. Activate the virtual environment
Linux
```bash
source venv/bin/activate
```
Windows
```bash
venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
python vits_fast_fine_tuning/monotonic_align/setup.py build_ext
```

### 4. Install ffmpeg
> [!NOTE]
> Please note that if you encounter an error indicating that ffmpeg cannot be found during training, as shown in the following error message, you need to install the ffmpeg library.
```bash
DEBUG:torio._extension.utils:Loading FFmpeg
DEBUG:torio._extension.utils:Failed to load FFmpeg extension.
```
According to the [official torch documentation](https://pytorch.org/audio/2.3.0/installation.html), the library is searched using the following naming conventions. If you encounter installation issues, check if the relevant files can be found.
> When searching for FFmpeg installation, TorchAudio looks for library files which have names with version numbers. That is, libavutil.so.<VERSION> for Linux, libavutil.<VERSION>.dylib for macOS, and avutil-<VERSION>.dll for Windows. Many public pre-built binaries follow this naming scheme, but some distributions have un-versioned file names. If you are having difficulties detecting FFmpeg, double check that the library files you installed follow this naming scheme, (and then make sure that they are in one of the directories listed in library search path.)

To install ffmpeg on Windows, you can find precompiled libraries on the following GitHub page:
https://github.com/BtbN/FFmpeg-Builds/releases

Make sure to download a version that includes the libraries. Since this project uses torchaudio, which requires ffmpeg versions 6, 5, or 4, it is recommended to download from the following link:
https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-n6.1-latest-win64-lgpl-shared-6.1.zip

## Launching the WebUI
> [!TIP]
> Most features can now be operated through the WebUI.

```bash
python webui.py
```

### User Interface
<div align="center">
  <img src="./docs/images/project-overview-1.png" alt="" width="600"/>
</div>

<div align="center">
  <img src="./docs/images/project-overview-2.png" alt="" width="600"/>
</div>

<div align="center">
  <img src="./docs/images/project-overview-3.png" alt="" width="600"/>
</div>

<div align="center">
  <img src="./docs/images/project-overview-4.png" alt="" width="600"/>
</div>

<div align="center">
  <img src="./docs/images/project-overview-5.png" alt="" width="600"/>
</div>

<div align="center">
  <img src="./docs/images/project-overview-6.png" alt="" width="600"/>
</div>

<div align="center">
  <img src="./docs/images/project-overview-7.png" alt="" width="600"/>
</div>

<div align="center">
  <img src="./docs/images/project-overview-8.png" alt="" width="600"/>
</div>

## Converting Pre-trained Models to Sherpa-Onnx Format
If you already have many pre-trained models and only want to use the conversion function, you can use the following script.

### 1. Run the script

```bash
python export_vits_fast_fine_tuning_onnx.py --config ./models/vits-xxx/config.json --checkpoint ./models/vits-xxx/G_latest.pth
```
Command Line Parameters
 - --config (required) The config file generated after VITS training.
 - --checkpoint (required) The PyTorch model generated after VITS training.
 - --output_dir (optional) The directory to output the ONNX model. The default is `onnx-output`.
 - --comment (optional) The comment information in the ONNX model.
 - --language (optional) The language information in the ONNX model.
 - --model_name (optional) The model name, mainly used for creating a subdirectory in the output folder.

### 2. Output Results
After successful output, you should find a folder named after the model name in the `onnx-output` directory. Inside this folder, you should see the following files and structure, which are used by Sherpa-ONNX. `model.onnx` is the original model, `model.int8.onnx` is the quantized model, and you can choose either one. `model-opt.onnx` is the optimized model, but it is currently uncertain whether Sherpa-ONNX supports it.
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

### 3. Inference
```bash
python onnx_inference.py --checkpoint ./onnx-output/vits-xxx/model.onnx --lexicon ./onnx-output/vits-xxx/lexicon.txt --tokens ./onnx-output/vits-xxx/tokens.txt
```
Command Line Parameters
 - --checkpoint (required) The ONNX model.
 - --lexicon (required) The phoneme table used by the model.
 - --token (required) The symbol table used by the model.
 - --text (optional) The content for text-to-speech.
