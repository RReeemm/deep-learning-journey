import os

root_dir = "PyTorch_learning\\hymenoptera_data\\train"   #the path of the dataset
target_dir = "bees_image"
img_path = os.listdir(os.path.join(root_dir,"bees_image"))  #get the image path
label_dir = target_dir.split("_")[0]    #get the label name
out_dir = "bees_label"
for i in img_path:
    file_name = i.split(".")[0] #get the file name
    with open(os.path.join(root_dir,out_dir,file_name+".txt"),"w") as f:    #create a txt file to save the label#
        f.write(label_dir)