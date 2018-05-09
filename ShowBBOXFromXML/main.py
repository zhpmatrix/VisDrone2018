import os
import cv2
import pickle
import argparse
import numpy as np
import pandas as pd
import xml.dom.minidom
import matplotlib.pyplot as plt
from PIL import Image,ImageDraw

root_dir = "../../data/Merge_VisDrone2018/"
ImgPath = root_dir+'JPEGImages/' 
AnnoPath = root_dir+'Annotations/'

def read_single_img(img_idx,bboxes):
    imgfile = ImgPath + img_idx+'.jpg' 
    xmlfile = AnnoPath + img_idx + '.xml'
        
    #img = Image.open(imgfile)
    DomTree = xml.dom.minidom.parse(xmlfile)
    annotation = DomTree.documentElement

    filenamelist = annotation.getElementsByTagName('filename')
    filename = filenamelist[0].childNodes[0].data
    objectlist = annotation.getElementsByTagName('object')
    
    for objects in objectlist:
        namelist = objects.getElementsByTagName('name')
        objectname = namelist[0].childNodes[0].data
        bndbox = objects.getElementsByTagName('bndbox')
        for box in bndbox:
            x1_list = box.getElementsByTagName('xmin')
            x1 = int(x1_list[0].childNodes[0].data)
            y1_list = box.getElementsByTagName('ymin')
            y1 = int(y1_list[0].childNodes[0].data)
            x2_list = box.getElementsByTagName('xmax')
            x2 = int(x2_list[0].childNodes[0].data)
            y2_list = box.getElementsByTagName('ymax')
            y2 = int(y2_list[0].childNodes[0].data)
            bboxes.setdefault(objectname,[]).append((x2 - x1, y2 - y1))
            # bboxes[objectname].append((x2-x1, y2-y1))
            # draw = ImageDraw.Draw(img)
            # print(x1,y1,x2,y2,img.size)
            # green color for rectangle
            # draw.rectangle([x1,y1,x2,y2],outline="green")
            # red color for text
            # draw.text((x1,y1),objectname,fill=(255,0,0,255))
    #img.show()

def read_all_images(train_list):
    bboxes = {}
    img_list = pd.read_csv(train_list, header=None)
    train_num = img_list.shape[0]
    for i in range( train_num):
        print(i,"/",img_list.shape[0])
        read_single_img(img_list.ix[i][0], bboxes)
    with open('bbox_dict.pkl', 'wb') as f:
        pickle.dump(bboxes, f)
        
    print('Done!')

def get_bbox():
    with open('bbox_dict.pkl', 'rb') as f:
        bboxes = pickle.load(f)
    for key in bboxes.keys():
        bbox = bboxes[key]
        plt.scatter([x[0] for x in bbox], [x[1] for x in bbox])
        plt.show()
        plt.close()

def get_mean_rgb(img_dir,img_size):
    img_list=os.listdir(img_dir)
    sum_r=0
    sum_g=0
    sum_b=0
    count=0

    for img_name in img_list:
        img_path=os.path.join(img_dir,img_name)
        img=cv2.imread(img_path)
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        img=cv2.resize(img,(img_size,img_size))
        
        sum_r=sum_r+img[:,:,0].mean()
        sum_g=sum_g+img[:,:,1].mean()
        sum_b=sum_b+img[:,:,2].mean()
        count=count+1
        print(count, "/", len(img_list))

    sum_r=sum_r/count
    sum_g=sum_g/count
    sum_b=sum_b/count
    img_mean=[sum_r,sum_g,sum_b]
    print(img_mean)

if __name__ == '__main__':
    
    #parser = argparse.ArgumentParser('main')
    #parser.add_argument('img_idx')
    #img_idx = '0000001_04527_d_0000008'
    #args = parser.parse_args()
    #read_single_img(args.img_idx)
    
    #train_list = "../../data/Merge_VisDrone2018/ImageSets/Main/train.txt"
    #read_all_images(train_list)
    #get_bbox()
    #print(bboxes.keys())

    #img_dir = "../../Data/VisDrone2018-DET-train/images/"
    #img_size = 1000
    #size=2000,RGB:95.003518834106174, 96.386738363931443, 92.885945304899025
    #get_mean_rgb(img_dir,img_size)


