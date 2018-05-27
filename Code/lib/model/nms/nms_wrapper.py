# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------
import torch
from model.utils.config import cfg

if torch.cuda.is_available():
    from model.nms.nms_gpu import nms_gpu
    #from model.nms.gpu_nms import gpu_nms
from model.nms.nms_cpu import nms_cpu
#from model.nms.cpu_nms import cpu_soft_nms

def nms(dets, thresh, force_cpu=False):
    """Dispatch to either CPU or GPU NMS implementations."""
    if dets.shape[0] == 0:
        return []
    # ---numpy version---
    # original: return gpu_nms(dets, thresh, device_id=cfg.GPU_ID)
    # ---pytorch version---

    return nms_gpu(dets, thresh) if force_cpu == False else nms_cpu(dets, thresh)
    #return gpu_nms(dets, thresh) if force_cpu == False else cpu_soft_nms(dets, thresh)
