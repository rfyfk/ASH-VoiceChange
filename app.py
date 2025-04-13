from assets.logging_config import configure_logging

configure_logging(True, False, "WARNING")

import sys
from typing import Any

import gradio as gr

from tabs.edge_tts import edge_tts_tab
from tabs.inference import inference_tab
from tabs.install import files_upload, install_hubert_tab, output_message, url_zip_download, zip_upload
from tabs.uvr import uvr_tab
from tabs.welcome import welcome_tab

DEFAULT_SERVER_NAME = "127.0.0.1"
DEFAULT_PORT = 4000
MAX_PORT_ATTEMPTS = 10

output_message_component = output_message()


def is_offline_mode() -> bool:
    return "--offline" in sys.argv


with gr.Blocks(
    title="ASH VoiceChange" if not is_offline_mode() else "ASH VoiceChange (offline)",
    css="footer{display:none !important}",
    theme=gr.themes.Soft(
        primary_hue="yellow",  # Оставление yellow для совместимости
    ).set(
        # Основной цвет (кнопки, акцентные элементы)
        button_primary_background_fill="#d9ba7e",
        button_primary_background_fill_dark="#d9ba7e",  # dark-mode
        button_primary_text_color="#e3e3e3",
        button_primary_border_color="#d9ba7e",
        
        # Цвета при наведении
        button_primary_background_fill_hover="#d9ba7e",
        slider_color="#d9ba7e",
        
        # Доп. элементы
        checkbox_background_color_selected="#d9ba7e",
        checkbox_border_color_selected="#d9ba7e",
        link_text_color="#d9ba7e",
    ),
) as VoiceChange:

    with gr.Tab("Контакты"):
        welcome_tab()

    with gr.Tab("VoiceChange"):
        inference_tab()

    if not is_offline_mode():
        with gr.Tab("TTS"):
            edge_tts_tab()

    with gr.Tab("UVR"):
        if is_offline_mode():
            gr.HTML(
                "<center><h3>UVR не будет функционировать без подключения к интернету, если вы ранее не установили необходимые модели.</h3></center>"
            )
        uvr_tab()

    with gr.Tab("Загрузка моделей"):
        if not is_offline_mode():
            with gr.Tab("Загрузка RVC моделей"):
                url_zip_download(output_message_component)
                zip_upload(output_message_component)
                files_upload(output_message_component)
                output_message_component.render()
            with gr.Tab("Загрузка HuBERT моделей"):
                install_hubert_tab()
        else:
            with gr.Tab("Загрузка RVC моделей"):
                zip_upload(output_message_component)
                files_upload(output_message_component)
                output_message_component.render()


def launch_gradio(server_name: str, server_port: int) -> None:
    VoiceChange.launch(
        favicon_path="assets/logo.ico",
        share="--share" in sys.argv,
        inbrowser="--open" in sys.argv,
        server_name=server_name,
        server_port=server_port,
        show_error=True,
    )


def get_value_from_args(key: str, default: Any = None) -> Any:
    if key in sys.argv:
        index = sys.argv.index(key) + 1
        if index < len(sys.argv):
            return sys.argv[index]
    return default


if __name__ == "__main__":
    port = int(get_value_from_args("--port", DEFAULT_PORT))
    server = get_value_from_args("--server-name", DEFAULT_SERVER_NAME)

    for _ in range(MAX_PORT_ATTEMPTS):
        try:
            launch_gradio(server, port)
            break
        except OSError:
            print(f"Не удалось запустить на порту {port}, повторите попытку на порту {port - 1}...")
            port -= 1
        except Exception as error:
            print(f"Произошла ошибка при запуске Gradio: {error}")
            break
