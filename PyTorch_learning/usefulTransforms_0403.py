from PIL import Image
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("logs")

img_path = "PyTorch_learning\\hymenoptera_data\\train\\ants_image\\0013035.jpg"
img_PIL = Image.open(img_path)

#totensor
trans_totensor = transforms.ToTensor()
img_tensor = trans_totensor(img_PIL)
writer.add_image("tensor_img", img_tensor, 1)


#normalize
print(img_tensor[0][0][0])
trans_norm = transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
img_norm = trans_norm(img_tensor)
print(img_norm[0][0][0])
writer.add_image("norm_tensor_img", img_norm, 1)

#resize
print(img_PIL.size)
trans_resize = transforms.Resize((224, 224))
img_resize = trans_resize(img_tensor)
writer.add_image("resize_img", img_resize, 1)

#compose
trans_resize_2 = transforms.Resize((512))
trans_compose = transforms.Compose([trans_resize_2,trans_resize])
img_compose = trans_compose(img_tensor)
writer.add_image("compose_img", img_compose, 1)

#randomcrop
trans_randomcrop = transforms.RandomCrop((224, 224))
trans_compose_2 = transforms.Compose([trans_totensor,trans_randomcrop])
for i in range(10):
    img_random = trans_compose_2(img_PIL)
    writer.add_image("randomcrop_img", img_random, i)


writer.close()