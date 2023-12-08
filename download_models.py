from tools.bot import ConversationBot

load = "VirtualTryon_cuda:0,ImageCaptioning_cuda:0,Text2Image_cuda:0,SDInpainting_cuda:0,ScribbleText2Image_cuda:0,CannyText2Image_cuda:0,Image2Canny_cpu,Image2Line_cpu,Image2Hed_cpu,Image2Scribble_cpu,Image2Pose_cpu,Image2Depth_cpu,Image2Normal_cpu"
bot = ConversationBot(load_dict=load)
bot.init_agent()
