import os
import re
import json
import uuid
import inspect
import numpy as np
import gradio as gr
from PIL import Image
from langchain.agents.tools import Tool
from langchain.llms.openai import OpenAI
from langchain.agents.initialize import initialize_agent
from langchain.chains.conversation.memory import ConversationBufferMemory

from .utils import rgb2rgba, cut_dialogue_history
from .consts import (
    FASHION_CHAT_PREFIX,
    FASHION_CHAT_FORMAT_INSTRUCTIONS,
    FASHION_CHAT_SUFFIX,
)
from .modules import (
    BackgroundRemoving,
    CannyText2Image,
    DalleGenerate,
    DalleInpainting,
    DepthText2Image,
    HedText2Image,
    ImageCaptioning,
    ImageEditing,
    Image2Canny,
    Image2Depth,
    Image2Hed,
    Image2Line,
    Image2Normal,
    Image2Pose,
    Image2Scribble,
    InfinityOutPainting,
    Inpainting,
    InstructPix2Pix,
    LineText2Image,
    NormalText2Image,
    ObjectSegmenting,
    PoseText2Image,
    ScribbleText2Image,
    SDInpainting,
    SDVideo,
    Segmenting,
    Text2Box,
    Text2Image,
    VisualQuestionAnswering,
    VirtualTryon,
)
from .modules.image_captioning import image_captioning


class ConversationBot:
    def __init__(self, load_dict):
        # load_dict = {'VisualQuestionAnswering':'cuda:0', 'ImageCaptioning':'cuda:1',...}
        load_dict = {
            e.split("_")[0].strip(): e.split("_")[1].strip()
            for e in load_dict.split(",")
        }
        self.image_captioning = image_captioning
        print(
            f"Initializing FashionChat,\nload_dict=\n{json.dumps(load_dict, indent=2)}"
        )
        if "ImageCaptioning" in load_dict:
            del load_dict["ImageCaptioning"]
            # raise ValueError(
            #     "You have to load ImageCaptioning as a basic function for FashionChat"
            # )

        self.models = {"ImageCaptioning": image_captioning}
        # Load Basic Foundation Models
        for class_name, device in load_dict.items():
            self.models[class_name] = globals()[class_name](device=device)

        # Load Template Foundation Models
        for class_name, module in globals().items():
            if getattr(module, "template_model", False):
                template_required_names = {
                    k
                    for k in inspect.signature(module.__init__).parameters.keys()
                    if k != "self"
                }
                loaded_names = set([type(e).__name__ for e in self.models.values()])
                if template_required_names.issubset(loaded_names):
                    self.models[class_name] = globals()[class_name](
                        **{name: self.models[name] for name in template_required_names}
                    )

        print(f"All the Available Functions: {self.models}")

        self.tools = []
        for instance in self.models.values():
            for e in dir(instance):
                if e.startswith("inference"):
                    func = getattr(instance, e)
                    self.tools.append(
                        Tool(name=func.name, description=func.description, func=func)
                    )
        self.llm = OpenAI(temperature=0)
        self.memory = ConversationBufferMemory(
            memory_key="chat_history", output_key="output"
        )

    def init_agent(self):
        self.memory.clear()  # clear previous history
        place = "Enter text and press enter, or upload an image"
        label_clear = "Clear Chat"
        self.agent = initialize_agent(
            self.tools,
            self.llm,
            agent="conversational-react-description",
            verbose=True,
            memory=self.memory,
            return_intermediate_steps=True,
            agent_kwargs={
                "prefix": FASHION_CHAT_PREFIX,
                "format_instructions": FASHION_CHAT_FORMAT_INSTRUCTIONS,
                "suffix": FASHION_CHAT_SUFFIX,
            },
        )
        return (
            gr.update(visible=True),
            gr.update(visible=False),
            gr.update(placeholder=place),
            gr.update(value=label_clear),
            gr.update(value="Male"),  # Gender
            gr.update(value="Cotton"),  # Fabric
            gr.update(value="T-Shirt"),  # Clothing Type
            gr.update(value="Black"),  # Color
            gr.update(value="Plain"),  # Pattern
        )

    def run_text(
        self,
        text,
        state,
        use_conditioning,
        gender,
        fabric,
        clothing_type,
        color,
        pattern,
    ):
        self.agent.memory.buffer = cut_dialogue_history(
            self.agent.memory.buffer, keep_last_n_words=500
        )
        if use_conditioning:
            text = f"{text}, a {clothing_type} for a {gender} made of {fabric} with {color} color and {pattern} pattern"
        res = self.agent({"input": text.strip()})
        res["output"] = res["output"].replace("\\", "/")
        response = re.sub(
            "(image/[-\w]*.png)",
            lambda m: f"![](file={m.group(0)})*{m.group(0)}*",
            res["output"],
        )
        state = state + [(text, response)]
        print(
            f"\nProcessed run_text, Input text: {text}\nCurrent state: {state}\n"
            f"Current Memory: {self.agent.memory.buffer}"
        )
        return state, state

    def run_image(self, image, state, txt):
        image_filename = os.path.join("image", f"{str(uuid.uuid4())[:8]}.png")
        print("======>Auto Resize Image...")
        img = Image.open(image.name)
        width, height = img.size
        ratio = min(512 / width, 512 / height)
        width_new, height_new = (round(width * ratio), round(height * ratio))
        width_new = int(np.round(width_new / 64.0)) * 64
        height_new = int(np.round(height_new / 64.0)) * 64
        img = img.resize((width_new, height_new))
        img = img.convert("RGB")
        img.save(image_filename, "PNG")
        print(f"Resize image form {width}x{height} to {width_new}x{height_new}")
        description = self.image_captioning.inference(image_filename)
        Human_prompt = f'\nHuman: provide a figure named {image_filename}. The description is: {description}. This information helps you to understand this image, but you should use tools to finish following tasks, rather than directly imagine from my description. If you understand, say "Received". \n'
        AI_prompt = "Received.  "
        self.agent.memory.buffer = (
            self.agent.memory.buffer + Human_prompt + "AI: " + AI_prompt
        )
        state = state + [(f"![](file={image_filename})*{image_filename}*", AI_prompt)]
        print(
            f"\nProcessed run_image, Input image: {image_filename}\nCurrent state: {state}\n"
            f"Current Memory: {self.agent.memory.buffer}"
        )
        return state, state, f"{image_filename} {txt}"

    def run_image_mask(self, inputs, state, txt):
        image_filename = os.path.join("image", f"{str(uuid.uuid4())[:8]}.png")
        mask_filename = image_filename.replace(".png", "_mask.png")
        rgba_filename = image_filename.replace(".png", "_rgba.png")
        print("======>Auto Resize Image...")
        img = inputs["image"]
        msk = inputs["mask"]
        width, height = img.size
        ratio = min(512 / width, 512 / height)
        width_new, height_new = (round(width * ratio), round(height * ratio))
        width_new = int(np.round(width_new / 64.0)) * 64
        height_new = int(np.round(height_new / 64.0)) * 64
        img = img.resize((width_new, height_new)).convert("RGB")
        msk = msk.resize((width_new, height_new)).convert("RGB")
        rgba = rgb2rgba(img, msk)
        img.save(image_filename, "PNG")
        msk.save(mask_filename, "PNG")
        rgba.save(rgba_filename, "PNG")
        print(f"Resize image form {width}x{height} to {width_new}x{height_new}")
        description = self.models["ImageCaptioning"].inference(image_filename)
        Human_prompt = f'\nHuman: provide a figure named {image_filename}. The description is: {description}. This information helps you to understand this image, but you should use tools to finish following tasks, rather than directly imagine from my description. If you understand, say "Received". \n'
        AI_prompt = "Received.  "
        self.agent.memory.buffer = (
            self.agent.memory.buffer + Human_prompt + "AI: " + AI_prompt
        )
        state = state + [(f"![](file={rgba_filename})*{image_filename}*", AI_prompt)]
        print(
            f"\nProcessed run_image, Input image: {image_filename}\nCurrent state: {state}\n"
            f"Current Memory: {self.agent.memory.buffer}"
        )
        return state, state, f"{image_filename} {txt}"
