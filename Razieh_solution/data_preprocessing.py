import torch
import torchvision
from torchvision.io import read_image
import torchvision.transforms as T
import numpy as np
import os
import random

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Get the list of image file names for three diseases
path_acne = "./data_3class_skin_diseases/acne/"
path_herpes = "./data_3class_skin_diseases/herpes_simplex/"
path_lichen = "./data_3class_skin_diseases/lichen_planus/"

acne_img_list = os.listdir(path_acne)
herpes_img_list = os.listdir(path_herpes)
lichen_img_list = os.listdir(path_lichen)

''' find best Height to width ration for images'''
def find_image_ratios(img_folders, folder_names = ["acne", "herpes_simplex", "lichen_planus"]):
    image_ratio = []
    width_list = []
    height_list = []
    vertical = 0
    for i in range(len(img_folders)):
        for _,path in enumerate(img_folders[i]):
            img = read_image("./data_3class_skin_diseases/"+folder_names[i]+"/"+path)
            width, height = min(img.shape[1], img.shape[2]), max(img.shape[1], img.shape[2])
            width_list.append(width)
            height_list.append(height)
            image_ratio.append(height/width)
            if img.shape[1]>img.shape[2]:
                vertical+=1

    print("Total number of images: ", len(image_ratio))
    print("Width of images: Mean = %i , Min = %i , Max = %i" %(np.mean(width_list), min(width_list), max(width_list)))
    print("Height of images: Mean = %i , Min = %i , Max = %i"%(np.mean(height_list), min(height_list), max(height_list)))
    print("H/W of images: Mean = %f , Min = %f , Max = %f" %(np.mean(image_ratio), min(image_ratio), max(image_ratio)))
    print("Number of vertical images = ", vertical)


# find_image_ratios([acne_img_list, herpes_img_list, lichen_img_list])

''' transforming data
Rotating vertical images
creating three set of images with dimensions of: 120*180, 400*600, 500*750 '''
def image_transformation(img_folders, folder_names, final_size):
    # Three transformers
    transform_1 = T.Resize(size = final_size[0])
    transform_2 = T.Resize(size = final_size[1])
    transform_3 = T.Resize(size = final_size[2])
    
    data_120_180 = False
    data_400_600 = False
    data_500_750 = False

    y_true = [0]
    for i in range(len(img_folders)):
        for _,path in enumerate(img_folders[i]):
            image = read_image("./data_3class_skin_diseases/"+folder_names[i]+"/"+path)
            if image.shape[1]>image.shape[2]:
                image = torch.transpose(image, 1, 2)
            
            image = torch.unsqueeze(image, 0)

            image_1 = transform_1(image)
            # image_2 = transform_2(image)
            # image_3 = transform_3(image)
            
            if data_120_180 is False:
                data_120_180 = image_1
                # data_400_600 = image_2
                # data_500_750 = image_3

            else:
                data_120_180 = torch.concat((data_120_180, image_1), 0)
                # data_400_600 = torch.concat((data_400_600, image_2), 0)
                # data_500_750 = torch.concat((data_500_750, image_3), 0)
                y_true.append(i)


    return data_120_180, data_400_600, data_500_750, y_true


def train_test_split(data, label):
    class1_idx = torch.from_numpy((np.where(torch.Tensor.numpy(label) == 0))[0])
    class2_idx = torch.from_numpy((np.where(torch.Tensor.numpy(label) == 1))[0])
    class3_idx = torch.from_numpy((np.where(torch.Tensor.numpy(label) == 2))[0])
    class1_train_idx, class1_val_idx, class1_test_idx =torch.tensor_split(class1_idx, (int(0.8*len(class1_idx)), int(0.9*len(class1_idx))))
    class2_train_idx, class2_val_idx, class2_test_idx =torch.tensor_split(class2_idx, (int(0.8*len(class2_idx)), int(0.9*len(class2_idx))))
    class3_train_idx, class3_val_idx, class3_test_idx =torch.tensor_split(class3_idx, (int(0.8*len(class3_idx)), int(0.9*len(class3_idx))))
    
    X_train = torch.cat((data[class1_train_idx],data[class2_train_idx],data[class3_train_idx],
                         data[class1_val_idx],data[class2_val_idx],data[class3_val_idx]), 0)
    X_test = torch.cat((data[class1_test_idx],data[class2_test_idx],data[class3_test_idx]), 0)
    
    y_train = torch.cat((label[class1_train_idx],label[class2_train_idx],label[class3_train_idx],
                         label[class1_val_idx],label[class2_val_idx],label[class3_val_idx]), 0)
    y_test = torch.cat((label[class1_test_idx],label[class2_test_idx],label[class3_test_idx]), 0)
    assert X_train.shape[0] == len(y_train)
    assert X_test.shape[0] == len(y_test)
    return X_train, X_test, y_train, y_test

''' Creating augmented images from original ones to have more train data'''
def data_augmentation(data, y_true, n=4, img_size=(120,180), rotate = True,
                      guassianblur = True,
                      colorjitter = True,
                      centercrop = True,
                      randomAutocontrast = True,
                      randomEqualize = True):
    random.seed(1111)
    np.random.seed(1111)
    new_data = data.clone().detach()
    new_y_true = torch.Tensor.tolist(y_true).copy()

    for i, image in enumerate(data):
        augmented_imgs = []
        if rotate:
            rotater = T.RandomRotation(degrees=(0, 180))
            [augmented_imgs.append(rotater(image)) for _ in range(n)]
            [new_y_true.append(y_true[i]) for _ in range(n)]

        if guassianblur:
            blure = T.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5))
            [augmented_imgs.append(blure(image)) for _ in range(n)]
            [new_y_true.append(y_true[i]) for _ in range(n)]

        if colorjitter:
            color_changer = T.ColorJitter(brightness=.5, hue=.3)
            [augmented_imgs.append(color_changer(image)) for _ in range(n)]
            [new_y_true.append(y_true[i]) for _ in range(n)]
        
        if centercrop:
            cropper = T.RandomCrop(size=img_size)
            [augmented_imgs.append(cropper(image)) for _ in range(n)]
            [new_y_true.append(y_true[i]) for _ in range(n)]
        
        if randomAutocontrast:
            contraster = T.RandomAutocontrast()
            [augmented_imgs.append(contraster(image)) for _ in range(n)]
            [new_y_true.append(y_true[i]) for _ in range(n)]
            

        if randomEqualize:
            equalizer = T.RandomEqualize()
            [augmented_imgs.append(equalizer(image)) for _ in range(n)]
            [new_y_true.append(y_true[i]) for _ in range(n)]
    
        for i in range(len(augmented_imgs)):
            new_data = torch.cat((new_data, torch.unsqueeze(augmented_imgs[i], 0)))

    return new_data, torch.FloatTensor(new_y_true)
    


def image_normalization(data):
    for image_list in [data]:
        image_list = image_list/255
        mean_data=image_list.mean()
        std_data = image_list.std()
        image_list = torchvision.transforms.functional.normalize(image_list, 
                                                                 [mean_data,mean_data,mean_data],
                                                                 [std_data,std_data,std_data])
    return image_list

# if __name__ == "__main__":
#     features = torch.load("./data_3class_skin_diseases/base_data_120_180.pt")
#     targets = torch.load("./data_3class_skin_diseases/base_labels.pt")

#     X_train,X_test, y_train,  y_test = train_test_split(features, targets)
    # torch.save(X_train, "./data/X_train_base.pt")
    # torch.save(X_test, "./data/X_test_base.pt")
    # torch.save(y_train, "./data/y_train_base.pt")
    # torch.save(y_test, "./data/y_test_base.pt")
    # X_train, y_train = data_augmentation(X_train, y_train)
    # X_test, y_test = data_augmentation(X_test, y_test)
    # torch.save(X_train, "./data/X_train_augmented.pt")
    # torch.save(y_train, "./data/y_train_augmented.pt")
    # torch.save(X_test, "./data/X_test_augmented.pt")
    # torch.save(y_test, "./data/y_test_augmented.pt")