import os
from PIL import Image,ImageDraw
import xml.dom.minidom

root_dir = "./VisDrone2018/"
ImgPath = root_dir+'JPEGImages/' 
AnnoPath = root_dir+'Annotations/'

imagelist = os.listdir(ImgPath)
for image in imagelist:
    image_pre, ext = os.path.splitext(image)
    imgfile = ImgPath + image 
    xmlfile = AnnoPath + image_pre + '.xml'
	
    img = Image.open(imgfile)
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
            draw = ImageDraw.Draw(img)
            
            # green color for rectangle
            draw.rectangle([x1,y1,x2,y2],outline="green")
            # red color for text
            draw.text((x1,y1),objectname,fill=(255,0,0,255))
    print(img.size)
    img.show()
