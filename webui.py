# Description: 建立功能的主頁面
# 第三方函式庫
import gradio as gr
# 本地函式庫
import logger
import ui
import ui.commons as commons

# 定義整個Gradio介面
def create_interface():
    with gr.Blocks(theme=commons.MyCustomTheme(), css=commons.css) as demo:
        lang = gr.Radio(choices=[("English", "en"), ("中文", "zh"), ("日本語", "ja")], label="Language")
        
        ui.create_youtube_download_interface(lang=lang)
        ui.create_denoise_audio_interface(lang=lang)
        ui.create_transcribe_audio_interface(lang=lang)
        ui.create_preprocess_interface(lang=lang)
        ui.create_finetuning_interface(lang=lang)
        ui.create_export_onnx_interface(lang=lang)
        ui.create_onnx_inference_interface(lang=lang)
        ui.create_stdout_interface(lang=lang)

    return demo

# 啟動Gradio介面
if __name__ == "__main__":
    demo = create_interface()
    # logger.sys_logger.info("Gradio interface launched.")
    print("Gradio interface launched.")
    demo.launch()