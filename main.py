import os
import gradio as gr
import argparse

from tools.consts import CSS
from tools import ConversationBot

if __name__ == "__main__":
    if not os.path.exists("checkpoints"):
        os.mkdir("checkpoints")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--load",
        type=str,
        default="VirtualTryon_cuda:0,ImageCaptioning_cuda:0,DalleInpainting_cuda:0,Text2Image_cuda:0,ScribbleText2Image_cuda:0,SDInpainting_cuda:0,Image2Canny_cpu,Image2Line_cpu,Image2Scribble_cpu,Image2Pose_cpu,DalleGenerate_cpu",
        # default="SDVideo_cuda:0",
    )
    args = parser.parse_args()
    bot = ConversationBot(load_dict=args.load)

    # with gr.Blocks(css="#chatbot .overflow-y-auto{height:500px}") as demo:
    with gr.Blocks(css=CSS) as demo:
        start = gr.Button("Load Agent")
        chatbot = gr.Chatbot(elem_id="chatbot", label="FashionChat")
        state = gr.State([])
        with gr.Row(visible=False) as inputs:
            with gr.Column(scale=0.7):
                txt = gr.Textbox(
                    show_label=False,
                    placeholder="Enter text and press enter, or upload an image",
                    container=False,
                )
            with gr.Column(scale=0.15, min_width=0):
                clear = gr.Button(value="Clear Chat", label="Clear Chat")
            with gr.Column(scale=0.15, min_width=0):
                btn = gr.UploadButton(label="Upload Image", file_types=["image"])
            with gr.Row():
                msk = gr.Image(source="upload", tool="sketch", type="pil")
                with gr.Column(scale=0.15, min_width=0):
                    create_msk = gr.Button("Create Mask")
        start.click(bot.init_agent, [], [inputs, start, txt, clear])
        txt.submit(bot.run_text, [txt, state], [chatbot, state])
        txt.submit(lambda: "", None, txt)
        btn.upload(bot.run_image, [btn, state, txt], [chatbot, state, txt])
        create_msk.click(bot.run_image_mask, [msk, state, txt], [chatbot, state, txt])
        clear.click(bot.memory.clear)
        clear.click(lambda: [], None, chatbot)
        clear.click(lambda: [], None, state)
    demo.launch(server_name="0.0.0.0", server_port=8080)
