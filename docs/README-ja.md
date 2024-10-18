<div align="center">
<h1>Sherpa-Onnx-VITS-Fine-Tuning-WebUI</h1>

VITSの推論、訓練、およびSherpa-Onnx形式での出力を統合したワンページWebUI

[![Static Badge](https://img.shields.io/badge/made_with-%F0%9F%92%96-red?style=for-the-badge&labelColor=orange)](https://github.com/anoneko777/sherpa-onnx-vits-fine-tuning-webui)
[![Open In Colab](https://img.shields.io/badge/Colab-F9AB00?style=for-the-badge&logo=googlecolab&color=525252)](https://github.com/anoneko777/sherpa-onnx-vits-fine-tuning-webui)
[![License](https://img.shields.io/badge/LICENSE-MIT-green.svg?style=for-the-badge)](https://github.com/anoneko777/sherpa-onnx-vits-fine-tuning-webui/blob/main/LICENSE)
[![Huggingface](https://img.shields.io/badge/🤗%20-online%20demo-yellow.svg?style=for-the-badge)](https://github.com/anoneko777/sherpa-onnx-vits-fine-tuning-webui)

[**English**](../README.md) | [**正體中文**](README-zh.md) | **日本語**

</div>

---

# 紹介
本プロジェクトは、簡単に操作できるワンページのWebUIを提供し、自分の音声ファイルを使用してVITSモデルを訓練し、推論結果をテストし、Sherpa-Onnxフレームワークで使用できるOnnxモデル形式で出力することができます。この技術は、Plachtaaが開発した[VITS-fast-fine-tuning](https://github.com/Plachtaa/VITS-fast-fine-tuning)と、Kaldiチームが開発した[Sherpa-Onnx](https://github.com/k2-fsa/sherpa-onnx)に基づいています。

本プロジェクトの開発目標は、自分の音声モデルをより簡単に訓練できるようにし、AndroidやiOSなどのモバイルプラットフォームで音声モデルを使用できるようにすることです。

> [!IMPORTANT]  
> 本プロジェクトはまだ開発段階にあるため、使用中にエラーやバグが発生する可能性があります。ご了承ください。ご連絡やissueの提供、解決策の提案を歓迎します。

## なぜVITSを使用するのか？
VITSはエンドツーエンドのテキスト音声合成（TTS）モデルであり、自然で感情豊かな音声合成を提供し、CPU推論でも優れたパフォーマンスを発揮します。本プロジェクトの立ち上げ時には、Bert-VITS2、GPT-SoVITS、FishSpeechなどのより優れたTTSモデルが存在しましたが、通常はGPU推論が必要であり、より多くのVRAMを消費します。VITS-fast-fine-tuningは、英語、中国語、日本語の音声訓練を提供します。

## なぜSherpa-Onnxを使用するのか？
Sherpa-Onnxフレームワークは、新世代のKaldiとonnxruntimeを使用して、音声からテキストへの変換、テキストから音声への変換、話者認識、VADを行い、インターネット接続を必要としません。組み込みシステム、Android、iOS、Raspberry Pi、RISC-V、x86_64サーバー、WebSocketサーバー/クライアント、C/C++、Python、Kotlin、C#、Go、NodeJS、Java、Swift、Dart、JavaScript、Flutter、Object Pascal、Lazarus、Rustをサポートします。したがって、VITSで訓練したモデル形式をSherpa-Onnxに変換することで、複数のプラットフォームでより効果的に使用できます。

# Todoリスト

- [ ] **機能:**
  - [x] 変換用のスクリプトを提供。
  - [ ] 操作可能なWebUIを提供。
  - [ ] Huggingface Online Demoを提供する。
  - [ ] Colabスクリプトを提供する。

# インストールと使用方法
> [!TIP]  
> 本プロジェクトはPython 3.8バージョンを使用しています。他のバージョンでは問題が発生する可能性があります。

## 事前訓練済みモデルをSherpa-Onnx形式に変換する
### 1. 事前訓練済みモデルをダウンロードする
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

### 2. 仮想環境を作成する
```bash
python3 -m venv venv
```

### 3. 仮想環境を起動する
Linux
```bash
source venv/bin/activate
```
Windows
```bash
venv\Scripts\activate
```

### 4. 依存関係をインストールする
```bash
pip install -r requirements.txt
python VITS-fast-fine-tuning/monotonic_align/setup.py build_ext
```

### 5. スクリプトを実行する

```bash
./export-vits-fast-fine-tuning-onnx.py --config ./models/vits-uma-genshin-honkai/config.json --checkpoint ./models/vits-uma-genshin-honkai/G_953000.pth
```

## WebUIを起動する
> [!NOTE]  
> 機能はまだ開発中です。

