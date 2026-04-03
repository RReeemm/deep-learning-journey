from torchvision import transforms
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
img_path = "PyTorch_learning\\hymenoptera_data\\train\\ants_image\\0013035.jpg"
img_PIL = Image.open(img_path)

writer = SummaryWriter("logs")

tensor_transforms = transforms.ToTensor()
img_tensor = tensor_transforms(img_PIL)

writer.add_image("tensor_img", img_tensor, 1)
writer.close()

