import os
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def bbox_iou(box1, box2):
    #get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_w1, b1_h1 = box1
    b1_x2, b1_y2 = b1_x1 + b1_w1 - 1, b1_y1 + b1_h1 -1
    
    b2_x1, b2_y1, b2_w2, b2_h2 = box2
    b2_x2, b2_y2 = b2_x1 + b2_w2 - 1, b2_y1 + b2_h2 -1
    
    inter_rect_x1 =  max(b1_x1, b2_x1)
    inter_rect_y1 =  max(b1_y1, b2_y1)
    inter_rect_x2 =  min(b1_x2, b2_x2)
    inter_rect_y2 =  min(b1_y2, b2_y2)
    
    inter_area = (inter_rect_x2 - inter_rect_x1 + 1)*(inter_rect_y2 - inter_rect_y1 + 1)
    
    b1_area = (b1_x2 - b1_x1 + 1)*(b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1)*(b2_y2 - b2_y1 + 1)
    
    if b1_area + b2_area - inter_area == 0:
        iou = 0.0
    else:
        iou = inter_area / (b1_area + b2_area - inter_area)
    return iou

def get_metric(real_root_annotations, pred_root_annotations, iou_threshold=0.50):
    ignored_region_idx = 0
    others_idx = 11
    truncation_ratio_threshold = 0.50   #0:,0.00;   1:0.01~0.50
    occlusion_ratio_threshold = 0.50    #0:0.00;    1:0.01~0.50;    2:0.50~1.00
    bbox_number_per_image = 500
    annotation_names = os.listdir(real_root_annotations)
    image_number = len(annotation_names)
    class_number = 10
    percision_all_classes = []
    for idx,real_annotations in enumerate(annotation_names):
        print(real_annotations,idx, '/', image_number)
        real = pd.read_csv(real_root_annotations+real_annotations, header=None, sep=',')
        pred = pd.read_csv(pred_root_annotations+real_annotations, header=None, sep=',')
        
        #choose ignored classes
        cate_idx = 5
        filtered_real_ignored_region = real[real.ix[:,cate_idx] != ignored_region_idx]
        filtered_real_others = filtered_real_ignored_region[filtered_real_ignored_region.ix[:,cate_idx] != others_idx]
        
        #filter occlusion(>0.50)
        #occlusion_idx = 7
        #filtered_real = filtered_real_others[filtered_real_others.ix[:,occlusion_idx] != 2] 

        filtered_real = filtered_real_others

        filtered_pred_ignored_region = pred[pred.ix[:,cate_idx] != ignored_region_idx]
        filtered_pred = filtered_pred_ignored_region[filtered_pred_ignored_region.ix[:,cate_idx] != others_idx]
        
        #sorted according to classification score
        score_idx = 4
        filtered_pred.sort_values(by=score_idx, ascending=False, inplace=True)
        
        #set number of bbox for each image as score 
        filtered_pred = filtered_pred[:bbox_number_per_image]
        
        percisions = [0 for cls_idx in range(class_number)]
        for cls_idx in range(class_number):
            pred_per_cls = filtered_pred[filtered_pred.ix[:,cate_idx] == cls_idx + 1]
            real_per_cls = filtered_real[filtered_real.ix[:,cate_idx] == cls_idx + 1]
            
            pred_per_cls.reset_index(drop=True,inplace=True)
            real_per_cls.reset_index(drop=True,inplace=True)
            
            real_cls_num = real_per_cls.shape[0]
            counter = 0 
            for i in range(pred_per_cls.shape[0]):
                for j in range(real_cls_num):
                    bbox1 = pred_per_cls.ix[i][:score_idx] #coordinate before score_idx
                    bbox2 = real_per_cls.ix[j][:score_idx]
                    iou = bbox_iou(bbox1, bbox2)
                    if iou > iou_threshold:
                       counter += 1
                       break
            if real_cls_num == 0:
                percisions[cls_idx] = 0 
            else:
                percisions[cls_idx] = counter * 1.0 / (real_cls_num) 
        #print(percisions)
        percision_all_classes.append(percisions)
    mAP = np.average( np.average(np.array(percision_all_classes), axis=0) )
    return mAP

def get_mAP(real_root_annotations, pred_root_annotations):
    iou_start,iou_end,iou_step = 0.50, 0.95, 0.05
    mmAP = []
    mAP_50 = 0.0
    mAP_75 = 0.0
    mAP_Range = 0.0

    for threshold in range(int( iou_start * 100) , int( iou_end * 100 ), int( iou_step * 100) ):
        iou_threshold =  threshold * 1.0 / 100
        mAP_ = get_metric(real_root_annotations, pred_root_annotations, iou_threshold)
        if iou_threshold == 0.50:
            mAP_50 = mAP_
        if iou_threshold == 0.75:
            mAP_75 = mAP_
        mmAP.append(mAP_)
    mAP_Range = sum(mmAP)*1.0/len(mmAP)
    return mAP_Range, mAP_50, mAP_75  
        
    
def show_det():
    pass

if __name__ == '__main__':
    epoch_num = 13
    iou_threshold = 0.75
    pred_root_annotations = '../../Results/Det/'+str(epoch_num)+'/annotations/'
    real_root_annotations = '../../Data/VisDrone2018-DET-val/annotations/'
    mAP_Range,mAP_50,mAP_75= get_mAP(real_root_annotations, pred_root_annotations)
    print('AP@0.50:0.95:',mAP_Range,'AP@0.50:',mAP_50,'AP@0.75:',mAP_75)
