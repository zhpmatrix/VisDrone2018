import os
import pandas as pd
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
    
    percisions_all_images = []
    for idx,real_annotations in enumerate(annotation_names):
        print(real_annotations,idx, '/', image_number)
        real = pd.read_csv(real_root_annotations+real_annotations, header=None, sep=',')
        pred = pd.read_csv(pred_root_annotations+real_annotations, header=None, sep=',')
        
        #choose ignored classes
        cate_idx = 5
        filtered_real_ignored_region = real[real.ix[:,cate_idx] != ignored_region_idx]
        filtered_real_others = filtered_real_ignored_region[filtered_real_ignored_region.ix[:,cate_idx] != others_idx]
        
        #filter occlusion(>0.50)
        occlusion_idx = 7
        filtered_real = filtered_real_others[filtered_real_others.ix[:,occlusion_idx] != 2] 

        filtered_pred_ignored_region = pred[pred.ix[:,cate_idx] != ignored_region_idx]
        filtered_pred = filtered_pred_ignored_region[filtered_pred_ignored_region.ix[:,cate_idx] != others_idx]
        
        #sorted according to classification score
        score_idx = 4
        filtered_pred.sort_values(by=score_idx, ascending=False, inplace=True)
        
        #set number of bbox for each image as score 
        filtered_pred = filtered_pred[:bbox_number_per_image]
        
        #calculate precision for each class in per image(IoU)
        cates = filtered_pred.ix[:,cate_idx].unique().tolist()
        percisions_per_image = []
        for cate in cates:
            cate_filtered_real = filtered_real[filtered_real.ix[:,cate_idx] == cate]
            cate_filtered_pred = filtered_pred[filtered_pred.ix[:,cate_idx] == cate]
            
            cate_filtered_real.reset_index(drop=True,inplace=True) 
            cate_filtered_pred.reset_index(drop=True,inplace=True) 
            
            #precision
            
            number_all_per_class = cate_filtered_real.shape[0]
            number_true_pos_per_class = 0
            for i in range(cate_filtered_pred.shape[0]):
                bbox1 = cate_filtered_pred.ix[i][:4].tolist()
                for j in range(number_all_per_class):
                    bbox2 = cate_filtered_real.ix[j][:4].tolist()
                    if bbox_iou(bbox1, bbox2) > iou_threshold:
                        number_true_pos_per_class += 1
                        #match only one real bbox for each pred bbox
                        break
            if number_all_per_class != 0:
                percision_per_class = ( number_true_pos_per_class * 1.0 ) / number_all_per_class
            else:
                percision_per_class = 0.0
            percisions_per_image.append(percision_per_class) 
        mAP_per_image = sum(percisions_per_image) * 1.0 / len(cates)
        print(real_annotations,mAP_per_image,percisions_per_image)
        percisions_all_images.append(mAP_per_image)
    mAP = sum(percisions_all_images) * 1.0 / image_number
    return 'mAP@'+str(iou_threshold)+':\t'+str(mAP)

def show_det():
    pass

if __name__ == '__main__':
    epoch_num = 13
    iou_threshold = 0.50
    pred_root_annotations = '../../Results/Det/'+str(epoch_num)+'/annotations/'
    real_root_annotations = '../../Data/VisDrone2018-DET-val/annotations/'
    mAP = get_metric(real_root_annotations, pred_root_annotations, iou_threshold)
    print(mAP)
