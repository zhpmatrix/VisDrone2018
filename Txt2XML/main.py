import os
from PIL import Image

root_dir = "VisDrone2018-DET-train/"
annotations_dir = root_dir+"annotations/"
image_dir = root_dir + "images/"
xml_dir = root_dir+"Annotations_XML/"
class_name = ['ignored regions','pedestrian','people','bicycle','car','van','truck','tricycle','awning-tricycle','bus','motor','others']

for filename in os.listdir(annotations_dir):
    fin = open(annotations_dir+filename, 'r')
    image_name = filename.split('.')[0]
    img = Image.open(image_dir+image_name+".jpg")
    xml_name = xml_dir+image_name+'.xml'
    with open(xml_name, 'w') as fout:
        fout.write('<annotation>'+'\n')
        
        fout.write('\t'+'<folder>VOC2007</folder>'+'\n')
        fout.write('\t'+'<filename>'+image_name+'.jpg'+'</filename>'+'\n')
        
        fout.write('\t'+'<source>'+'\n')
        fout.write('\t\t'+'<database>'+'VisDrone2018 Database'+'</database>'+'\n')
        fout.write('\t\t'+'<annotation>'+'VisDrone2018'+'</annotation>'+'\n')
        fout.write('\t\t'+'<image>'+'flickr'+'</image>'+'\n')
        fout.write('\t\t'+'<flickrid>'+'Unspecified'+'</flickrid>'+'\n')
        fout.write('\t'+'</source>'+'\n')
        
        fout.write('\t'+'<owner>'+'\n')
        fout.write('\t\t'+'<flickrid>'+'Haipeng Zhang'+'</flickrid>'+'\n')
        fout.write('\t\t'+'<name>'+'Haipeng Zhang'+'</name>'+'\n')
        fout.write('\t'+'</owner>'+'\n')
        
        fout.write('\t'+'<size>'+'\n')
        fout.write('\t\t'+'<width>'+str(img.size[0])+'</width>'+'\n')
        fout.write('\t\t'+'<height>'+str(img.size[1])+'</height>'+'\n')
        fout.write('\t\t'+'<depth>'+'3'+'</depth>'+'\n')
        fout.write('\t'+'</size>'+'\n')
        
        fout.write('\t'+'<segmented>'+'0'+'</segmented>'+'\n')

        for line in fin.readlines():
            line = line.split(',')
            fout.write('\t'+'<object>'+'\n')
            fout.write('\t\t'+'<name>'+class_name[int(line[5])]+'</name>'+'\n')
            fout.write('\t\t'+'<pose>'+'Unspecified'+'</pose>'+'\n')
            fout.write('\t\t'+'<truncated>'+line[6]+'</truncated>'+'\n')
            fout.write('\t\t'+'<difficult>'+str(int(line[7]))+'</difficult>'+'\n')
            fout.write('\t\t'+'<bndbox>'+'\n')
            fout.write('\t\t\t'+'<xmin>'+line[0]+'</xmin>'+'\n')
            fout.write('\t\t\t'+'<ymin>'+line[1]+'</ymin>'+'\n')
            fout.write('\t\t\t'+'<xmax>'+str(int(line[0])+int(line[2]))+'</xmax>'+'\n')
            fout.write('\t\t\t'+'<ymax>'+str(int(line[1])+int(line[3]))+'</ymax>'+'\n')
            fout.write('\t\t'+'</bndbox>'+'\n')
            fout.write('\t'+'</object>'+'\n')
             
        fin.close()
        fout.write('</annotation>')
