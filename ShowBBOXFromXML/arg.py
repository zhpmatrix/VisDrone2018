import os
import cv2
import pickle
import argparse
import numpy as np
import pandas as pd
import xml.dom.minidom
import matplotlib.pyplot as plt
from PIL import Image,ImageDraw

def read_single_img(img_idx):
    imgfile = img_idx +'.jpg' 
    xmlfile = img_idx + '.xml'
    img = Image.open(imgfile)
    DomTree = xml.dom.minidom.parse(xmlfile)
    annotation = DomTree.documentElement

    filenamelist = annotation.getElementsByTagName('filename')
    filename = filenamelist[0].childNodes[0].data
    objectlist = annotation.getElementsByTagName('object')
    #bboxes.append(len(objectlist))    
    for objects in objectlist:
        namelist = objects.getElementsByTagName('name')
        objectname = namelist[0].childNodes[0].data
        bndbox = objects.getElementsByTagName('bndbox')
        for box in bndbox:
            if objectname == 'awning-tricycle':
                x1_list = box.getElementsByTagName('xmin')
                x1 = int(x1_list[0].childNodes[0].data)
                y1_list = box.getElementsByTagName('ymin')
                y1 = int(y1_list[0].childNodes[0].data)
                x2_list = box.getElementsByTagName('xmax')
                x2 = int(x2_list[0].childNodes[0].data)
                y2_list = box.getElementsByTagName('ymax')
                y2 = int(y2_list[0].childNodes[0].data)
                  
                draw = ImageDraw.Draw(img)
                # green color for rectangle
                draw.rectangle([x1,y1,x2,y2],outline="green")
                # red color for text
                draw.text((x1,y1),objectname,fill=(255,0,0,255))
    img.show()

if __name__ == '__main__':
    
    img_idx = '0000068_00460_d_0000002'
    read_single_img(img_idx)
    
