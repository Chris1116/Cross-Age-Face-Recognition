import csv
import os
from PIL import Image
import shutil
from torchvision import transforms
from torchvision.transforms import functional as TF
import random
import torchvision
import matplotlib.pyplot as plt
import numpy as np

img_size = 224

def normalize(img_pil):
    m_0 = random.uniform(0.5, 1)
    m_1 = random.uniform(0.5, 1)
    m_2 = random.uniform(0.5, 1)
    s_0 = random.uniform(0.3, 1)
    s_1 = random.uniform(0.3, 1)
    s_2 = random.uniform(0.3, 1)
    mean = [m_0, m_1, m_2]
    std = [s_0, s_1, s_2]
    transform = transforms.Compose([
        transforms.Resize((img_size,img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std), 
        transforms.ToPILImage() 
    ])

    img_pil_normal = transform(img_pil)
    
    return img_pil_normal



def center_crop(img_pil):
    transform = transforms.Compose([
        transforms.CenterCrop(img_size),
    ])
    new_img = transform(img_pil)
    np.array(new_img).shape
    return new_img

def padding_black(img_pil):
    m_0 = random.randint(5, 40)
    m_1 = random.randint(5, 40)
    m_2 = random.randint(5, 40)
    m_3 = random.randint(5, 40)
    padding = (m_0, m_1, m_2, m_3)
    transform = transforms.Compose([
        transforms.Resize((img_size-m_1-m_3,img_size-m_0-m_2)),
        transforms.Pad(padding, fill=0,padding_mode="constant"), 
    ])
    new_img = transform(img_pil)
    #print(np.array(new_img).shape)
    return new_img

def padding_color(img_pil):
    padding = random.randint(10, 40)
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)
    transform = transforms.Compose([
        transforms.Resize((img_size-2*padding,img_size-2*padding)),
        transforms.Pad(padding, fill=(r,g, b),padding_mode="constant"), 
    ])
    new_img = transform(img_pil)
    #print(np.array(new_img).shape)
    return new_img

def padding_edge(img_pil):  
    m_0 = random.randint(5, 40)
    m_1 = random.randint(5, 40)
    m_2 = random.randint(5, 40)
    m_3 = random.randint(5, 40)
    padding = (m_0, m_1, m_2, m_3)
    transform = transforms.Compose([
        transforms.Resize((img_size-m_1-m_3,img_size-m_0-m_2)),
        transforms.Pad(padding, padding_mode="edge"), 
    ])
    new_img = transform(img_pil)
    #print(np.array(new_img).shape)

    return new_img

def padding_symmetric(img_pil):
    m_0 = random.randint(5, 40)
    m_1 = random.randint(5, 40)
    m_2 = random.randint(5, 40)
    m_3 = random.randint(5, 40)
    padding = (m_0, m_1, m_2, m_3)
    transform = transforms.Compose([
        transforms.Resize((img_size-m_1-m_3,img_size-m_0-m_2)),
        transforms.Pad(padding, padding_mode="symmetric"), 
    ])
    new_img = transform(img_pil)
    return new_img

def randcrop(img_pil):
    size=(img_size, img_size)
    transform = transforms.Compose([
        transforms.RandomCrop(size)
    ])
    new_img = transform(img_pil)
    return new_img

def Hflip(img_pil):
    transform = transforms.Compose([
        transforms.Resize((img_size,img_size)),
        transforms.RandomHorizontalFlip(p=0.95),
    ])

    new_img = transform(img_pil)
    return new_img

def Vflip(img_pil):
    transform = transforms.Compose([
        transforms.Resize((img_size,img_size)),
        transforms.RandomVerticalFlip(p=0.05),
    ])

    new_img = transform(img_pil)
    return new_img

def blur(img_pil):
    transform = transforms.Compose([
        transforms.Resize((img_size,img_size)),
        transforms.GaussianBlur(3,15)
    ])
    new_img = transform(img_pil)
    return new_img

def randAffine(img_pil):
    transform = transforms.Compose([
        transforms.Resize((img_size,img_size)),
        transforms.RandomAffine(degrees=(-15,15), translate=(0, 0.5), scale=(0.4, 0.7), shear=(0,0))
    ])

    new_img = transform(img_pil)
    return new_img

def to_gray(img_pil):
    transform = transforms.Compose([
        transforms.Resize((img_size,img_size)),
        transforms.Grayscale(num_output_channels=3)
    ])
    new_img = transform(img_pil)
    new_img_array = np.array(new_img)
    #print("original shape:", np.array(img_pil).shape)
    #print("shape:", new_img_array.shape)
    return new_img

def colorJitter(img_pil):
    transform = transforms.Compose([
        transforms.Resize((img_size,img_size)),
        transforms.ColorJitter(brightness=(1, 3), contrast=(
            1, 5), saturation=(1, 5), hue=(-0.1, 0.1))
    ])

    new_img = transform(img_pil)
    return new_img




'''
imagepath='./calfw/aligned_images/AJ_Cook_0001.jpg'

    
# read image with PIL module
img_pil = Image.open(imagepath, mode='r')
img_pil = img_pil.convert('RGB')
img_pil
'''
os.makedirs('data/calfw', exist_ok=True)
os.makedirs('data/calfw/train', exist_ok=True)

with open('CALFW_trainlist.csv', newline='') as csvfile:

    rows = csv.reader(csvfile)

    for row in rows:
        fdir = os.makedirs('./data/calfw/train/' + row[1], exist_ok=True)
        src_path = 'calfw/aligned_images/' + row[0]
        dst_path = './data/calfw/train/' + row[1] + '/' + row[0]
        img_name = row[0][:-4]

        # original train image
        img_pil = Image.open(src_path).resize((img_size, img_size))
        img_pil = to_gray(img_pil)
        img_pil.save(dst_path)

        #augmentatin train image
        dst_path = './data/calfw/train/' + row[1] + '/' + img_name

        
        normalize(img_pil).save(dst_path + '1.jpg')
        center_crop(img_pil).save(dst_path + '2.jpg')
        padding_black(img_pil).save(dst_path + '3.jpg')
        padding_color(img_pil).save(dst_path + '4.jpg')
        padding_edge(img_pil).save(dst_path + '5.jpg')
        padding_symmetric(img_pil).save(dst_path + '6.jpg')
        randcrop(img_pil).save(dst_path + '7.jpg')
        randcrop(img_pil).save(dst_path + '8.jpg')
        
        out = Hflip(img_pil)
        randcrop(out).save(dst_path + '9.jpg')
        
        Hflip(img_pil).save(dst_path + '10.jpg')
        #Vflip(img_pil).save(dst_path + '11.jpg')
        blur(img_pil).save(dst_path + '12.jpg')
        
        out = blur(img_pil)
        randcrop(out).save(dst_path + '13.jpg')
        
        randAffine(img_pil).save(dst_path + '14.jpg')
        
        out = randAffine(img_pil)
        randcrop(out).save(dst_path + '15.jpg')
        
        #to_gray(img_pil).save(dst_path + '16.jpg')
        
        '''
        colorJitter(img_pil).save(dst_path + '17.jpg')
        colorJitter(img_pil).save(dst_path + '18.jpg')
        colorJitter(img_pil).save(dst_path + '19.jpg')
       
        
        out = colorJitter(img_pil)
        randcrop(out).save(dst_path + '20.jpg')
        
       
        out = colorJitter(img_pil)
        randAffine(out).save(dst_path + '21.jpg')
        
        out = colorJitter(img_pil)
        out = Hflip(out)
        randAffine(out).save(dst_path + '22.jpg')
        
        out = colorJitter(img_pil)
        out = randAffine(out)
        randcrop(out).save(dst_path + '23.jpg')
        
        out = colorJitter(img_pil)
        out = Hflip(out)
        Hflip(img_pil).save(dst_path + '24.jpg')
        '''

os.makedirs('data/calfw/val', exist_ok=True)

with open('CALFW_validationlist.csv', newline='') as csvfile:

    rows = csv.reader(csvfile)

    for row in rows:
      fdir = os.makedirs('./data/calfw/val/' + row[1], exist_ok=True)
      src_path = 'calfw/aligned_images/' + row[0]
      dst_path = './data/calfw/val/' + row[1] + '/' + row[0]
      im = Image.open(src_path).resize((img_size, img_size)).save(dst_path)
      
