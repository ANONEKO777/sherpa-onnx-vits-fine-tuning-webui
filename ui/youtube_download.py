import os
import platform
from concurrent.futures import ThreadPoolExecutor
import random
# 第三方函式庫
import gradio as gr
import gradio_i18n
from gradio_i18n import translate_blocks
import pandas as pd
# 本地函式庫
from .commons import add_prefix_to_translations, gettext, set_page_prefix

_raw_audio_folder = os.path.join("training_data", "raw_audio")

_translations = {
    "en": {
        "title": "<h1 style='text-align: center;'>YouTube Download Page</h1>",
        "Speaker": "Speaker",
        "YouTube URL": "YouTube URL",
        "Add to List": "Add to List",
        "Video List": "Video List",
        "Delete Index": "Delete Index",
        "Delete from List": "Delete from List",
        "Download All": "Download All",
        "Download Result": "Download Result",
        "Download complete!": "Download complete!",
    },
    "zh": {
        "title": "<h1 style='text-align: center;'>YouTube下載頁面</h1>",
        "Speaker": "說話者",
        "YouTube URL": "YouTube URL",
        "Add to List": "加入列表",
        "Video List": "影片列表",
        "Delete Index": "刪除索引",
        "Delete from List": "從列表中刪除",
        "Download All": "全部下載",
        "Download Result": "下載結果",
        "Download complete!": "下載完成！",
    },
    "ja": {
        "title": "<h1 style='text-align: center;'>YouTubeダウンロードページ</h1>",
        "Speaker": "話者",
        "YouTube URL": "YouTube URL",
        "Add to List": "リストに追加",
        "Video List": "ビデオリスト",
        "Delete Index": "インデックスを削除",
        "Delete from List": "リストから削除",
        "Download All": "すべてダウンロード",
        "Download Result": "ダウンロード結果",
        "Download complete!": "ダウンロード完了！",
    },
}

_page_prefix = "youtube_download"
# 使用函數為每個鍵增加前綴，原因是目前i18n套件的字典是全介面共用的，因此用前綴區分不同功能頁面的翻譯
_translations = add_prefix_to_translations(_translations, _page_prefix)

# 初始化一個空的 DataFrame
df = pd.DataFrame(columns=["speaker", "url"])

def add_to_list(speaker, url):
    global df
    # 判斷url是否已經存在於df中，或者是否為空
    if url in df["url"].values or speaker == "" or url == "":
        return df
    new_row = {"speaker": speaker, "url": url}
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    return df

def delete_from_list(index):
    global df
    df = df.drop(int(index)).reset_index(drop=True)
    return df

def df_select_callback(df: pd.DataFrame, evt: gr.SelectData):
    seletected_index = evt.index
    # 返回被事件選中的row的index
    return seletected_index[0]

def generate_infos(video_df):
    infos = []
    for index, row in video_df.iterrows():
        speaker = row["speaker"]
        filename = speaker + "_" + str(random.randint(0, 1000000))
        infos.append({"link": row["url"], "filename": filename})
    return infos

def download_video(info):
    link = info["link"]
    filename = info["filename"]
    # linux和windows皆使用yt-dlp下載音訊
    if platform.system() == "Linux" or platform.system() == "Windows":
        os.system(f"yt-dlp {link} -x --audio-format wav --output {_raw_audio_folder}/{filename}.wav --no-check-certificate")
    # macos下使用yt-dlp_macos
    elif platform.system() == "Darwin":
        os.system(f"yt-dlp_macos {link} -x --audio-format wav --output {_raw_audio_folder}/{filename}.wav --no-check-certificate")

def download_videos(info):
    with set_page_prefix(_page_prefix):
        os.makedirs(_raw_audio_folder, exist_ok=True)
        try:
            infos = generate_infos(df)
            print(infos)
            with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
                executor.map(download_video, infos)
        except Exception as e:
            return str(e)

        return gettext("Download complete!")

def create_youtube_download_interface(lang):
    with set_page_prefix(_page_prefix):
        with gr.Blocks() as youtube_download_blocks:
            title = gr.Markdown(gettext("title"))

            with gr.Row():
                speaker_input = gr.Textbox(label=gettext("Speaker"))
                url_input = gr.Textbox(label=gettext("YouTube URL"))

            add_button = gr.Button(value=gettext("Add to List"))
            video_df = gr.DataFrame(df, label=gettext("Video List"), interactive=True)
            with gr.Row():
                delete_index = gr.Textbox(label=gettext("Delete Index"))
                delete_button = gr.Button(value=gettext("Delete from List"))
            download_button = gr.Button(value=gettext("Download All"))
            download_result = gr.Textbox(label=gettext("Download Result"))

            add_button.click(fn=add_to_list, inputs=[speaker_input, url_input], outputs=video_df)
            download_button.click(fn=download_videos, outputs=download_result)
            video_df.select(fn=df_select_callback, inputs=[video_df], outputs=delete_index)
            
            delete_button.click(delete_from_list, inputs=delete_index, outputs=video_df)
    
    translate_blocks(block=youtube_download_blocks, translation=_translations, lang=lang)