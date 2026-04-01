from torch.utils.tensorboard import SummaryWriter
import numpy as np
from PIL import Image

writer = SummaryWriter("logs")
image_path = "Python\\study\\PyTorch_learning\\hymenoptera_data\\train\\ants_image\\5650366_e22b7e1065.jpgg"
img_PIL = Image.open(image_path)
img_array = np.array(img_PIL)
writer.add_image("test", img_array, 1, dataformats="HWC")


for i in range(100):
    writer.add_scalar("y=2x", 3 * i, i)
writer.close()