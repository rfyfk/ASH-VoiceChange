import gradio as gr


def welcome_tab():
    gr.HTML(
        """
    <center>
        <h1 style="font-size: 3em;">
            <b>VoiceChange</b>
        </h1>
    </center>
    """
    )
    with gr.Row():
        with gr.Column(variant="panel"):
            gr.HTML(
                "<center><h2>"
                "<a href='https://t.me/user_rfyfk'>"  # Ссылка
                "Telegram"  # Имя ссылки
                "</a></h2></center>"
            )
            gr.HTML(
                "<center><h2>"
                "<a href='https://vk.com/user.rfyfk'>"  # Ссылка
                "ВКонтакте"  # Имя ссылки
                "</a></h2></center>"
            )
        with gr.Column(variant="panel"):
            gr.HTML(
                "<center><h2>"
                "<a href='https://t.me/pol1trees'>"  # Ссылка
                "ASH Telegram"  # Имя ссылки
                "</a></h2></center>"
            )
            gr.HTML(
                "<center><h2>"
                "<a href='https://t.me/+GMTP7hZqY0E4OGRi'>"  # Ссылка
                "ASH Chat"  # Имя ссылки
                "</a></h2></center>"
            )
