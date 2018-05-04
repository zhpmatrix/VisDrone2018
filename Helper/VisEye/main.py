import os
import pandas as pd
import matplotlib.pyplot as plt

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

if __name__ == '__main__':
    real_root_annotations = '../../../Data/VisDrone2018-DET-val/annotations/'   get_scale(real_root_annotations)
