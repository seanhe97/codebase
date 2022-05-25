import json
from tqdm import tqdm

import torch
import torch.nn.functional as F

from refer_python3 import REFER


def grounding_eval(results,dets,cocos,refer,alpha,mask_size=24):
    
    correct_A_d, correct_B_d, correct_val_d = 0, 0, 0
    correct_A, correct_B, correct_val = 0, 0, 0 
    num_A,num_B,num_val = 0,0,0
    
    for res in tqdm(results):

        ref_id = res['ref_id']
        ref = refer.Refs[ref_id]
        ref_box = refer.refToAnn[ref_id]['bbox']
        image = refer.Imgs[ref['image_id']]

        mask = res['pred'].cuda().view(1,1,mask_size,mask_size)    
        mask = F.interpolate(mask,size = (image['height'],image['width']), mode='bicubic').squeeze()
        
        # rank detection boxes
        max_score = 0
        for det in dets[str(ref['image_id'])]:
            score = mask[int(det[1]):int(det[1]+det[3]),int(det[0]):int(det[0]+det[2])]
            area = det[2]*det[3]
            score = score.sum() / area**alpha
            if score>max_score:
                pred_box = det[:4]
                max_score = score    

        IoU_det = computeIoU(ref_box, pred_box)
        
        if ref['split']=='testA':
            num_A += 1    
            if IoU_det >= 0.5:   
                correct_A_d += 1            
        elif ref['split']=='testB':
            num_B += 1    
            if IoU_det >= 0.5:   
                correct_B_d += 1    
        elif ref['split']=='val':
            num_val += 1    
            if IoU_det >= 0.5:   
                correct_val_d += 1    
                
    eval_result = {'val_d':correct_val_d/num_val,'testA_d':correct_A_d/num_A,'testB_d':correct_B_d/num_B}        
    
    for metric, acc in eval_result.items():
        print(f'{metric}: {acc:.3f}')
        
    return eval_result    



# IoU function
def computeIoU(box1, box2):
    # each box is of [x1, y1, w, h]
    inter_x1 = max(box1[0], box2[0])
    inter_y1 = max(box1[1], box2[1])
    inter_x2 = min(box1[0]+box1[2]-1, box2[0]+box2[2]-1)
    inter_y2 = min(box1[1]+box1[3]-1, box2[1]+box2[3]-1)

    if inter_x1 < inter_x2 and inter_y1 < inter_y2:
        inter = (inter_x2-inter_x1+1)*(inter_y2-inter_y1+1)
    else:
        inter = 0
    union = box1[2]*box1[3] + box2[2]*box2[3] - inter
    return float(inter)/union

if __name__ == "__main__":

    refcoco_data_path = '/youtu/xlab-team1/sunanhe/data/albef/data'
    det_file_path = '/youtu/xlab-team1/sunanhe/data/albef/data/refcoco+/dets.json'
    coco_file_path = '/youtu/xlab-team1/sunanhe/data/albef/data/refcoco+/cocos.json'

    refer = REFER(refcoco_data_path, 'refcoco+', 'unc')
    dets = json.load(open(det_file_path,'r'))
    cocos = json.load(open(coco_file_path,'r'))    

    B, H, W = 16, 24, 24
    results = torch.randn((B, H, W))

    grounding_acc = grounding_eval(results, dets, cocos, refer, alpha=0.5, mask_size=24)

    log_stats = {**{f'{k}': v for k, v in grounding_acc.items()}}