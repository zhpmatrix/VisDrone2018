import os
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

def get_scale(ann_dir):
    """
        show the scale of image including width and height.
        CONCLUSION: most of the objects are small and medium(< 96 as COCO)
    """
    for filename in os.listdir(ann_dir):
        print(filename)
        data = pd.read_csv(ann_dir + filename, header=None, sep=',')
        plt.scatter(data.ix[:,2], data.ix[:,3])
        plt.pause(1)
        plt.close()

def get_wh(img_dir):
    size_set = {}
    counter = 0
    for filename in os.listdir(img_dir):
        counter += 1
        img = Image.open(img_dir+filename)
        print(counter,filename,img.size)
        size_set.setdefault(img.size,[]).append(filename)
    for key in size_set.keys():
        print(key, len(size_set[key]) )


if __name__ == '__main__':
    
    #real_root_annotations = '../../../Data/VisDrone2018-DET-val/annotations/'   
    #get_scale(real_root_annotations)
    
    real_root_images = '../../Data/VisDrone2018-DET-val/images/'   
    """
    TRAIN:

    (1398, 1048), (1344, 756), (1920, 1080), 
    (1360, 765), (1916, 1078), (1400, 1050), 
    (2000, 1500), (1389, 1042), (1400, 788), 
    (480, 360), (960, 540)
    
    VAL:
    (1920, 1080), (1360, 765), (960, 540)
    
    TEST:
    (960, 540), (1920, 1080), (1360, 765),
    (1916, 1078), (1400, 1050), (1400, 788)

    """
    get_wh(real_root_images)
