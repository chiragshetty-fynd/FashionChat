from fashion_chat import ConversationBot

load = "ImageCaptioning_cuda:0,Text2Image_cuda:0,ScribbleText2Image_cuda:0,CannyText2Image_cuda:0,Image2Canny_cpu,Image2Line_cpu,Image2Hed_cpu,Image2Scribble_cpu,Image2Pose_cpu,Image2Depth_cpu,Image2Normal_cpu"
load_dict = {e.split("_")[0].strip(): e.split("_")[1].strip() for e in load.split(",")}
bot = ConversationBot(load_dict=load_dict)
bot.init_agent("English")
