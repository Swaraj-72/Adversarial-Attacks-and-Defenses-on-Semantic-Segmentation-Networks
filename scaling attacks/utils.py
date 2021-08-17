import torch
import numpy as np

def decode_segmap(image, nc=21): # visualise the decoded prediction (pred)
  
  label_colors = np.array([(0, 0, 0),
               (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128),
               (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0),
               (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128),
               (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128)])
  r = torch.zeros_like(image, dtype = torch.uint8)
  g = torch.zeros_like(image, dtype = torch.uint8)
  b = torch.zeros_like(image, dtype = torch.uint8)
  
  for l in range(0, nc):
    idx = image == l
    r[idx] = label_colors[l, 0]
    g[idx] = label_colors[l, 1]
    b[idx] = label_colors[l, 2]
    
  rgb = torch.stack([r, g, b], axis=3)
  return rgb

class_names = {0:'background', 1:'aeroplane', 2:'bicycle', 3:'bird', 4:'boat', 5:'bottle', 6:'bus', 7:'car', 8:'cat',
               9:'chair', 10:'cow', 11:'diningtable', 12:'dog', 13:'horse', 14:'motorbike', 15:'person', 16:'pottedplant',
               17:'sheep', 18:'sofa', 19:'train', 20:'tvmonitor' }


def IoUAcc(y_trg, y_pred, class_names = class_names):

    trg = y_trg.squeeze(1)
    pred = y_pred
    iou_list = list()
    present_iou_list = list()

    pred = pred.view(-1)
    trg = trg.view(-1)
    for sem_class in range(21): # loop over each class for IoU calculation
        pred_inds = (pred == sem_class)
        target_inds = (trg == sem_class)
        if target_inds.long().sum().item() == 0:
            iou_now = float('nan')
            #print('Class {} IoU is {}'.format(class_names[sem_class+1], iou_now))
        else: 
            intersection_now = (pred_inds[target_inds]).long().sum().item()
            union_now = pred_inds.long().sum().item() + target_inds.long().sum().item() - intersection_now
            iou_now = float(intersection_now) / float(union_now)
            #print('Class {} IoU is {}'.format(class_names[sem_class+1], iou_now))
            present_iou_list.append(iou_now)
        iou_list.append(iou_now)
        
    acc = accuracy_score(trg, pred)
    
    #print('mIoU is {} and Pixel Accuracy is {}'.format(np.mean(present_iou_list)*100, acc*100))
    return np.mean(present_iou_list)*100, acc*100
  
  
def result_attack_image(src_img, tar_img): # Function to apply Image Scaling Attack
    
    inv_normalize = T.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],std=[1/0.229, 1/0.224, 1/0.225])
    src_img, tar_img = inv_normalize(src_img), inv_normalize(tar_img)
    
    low = torch.nn.functional.interpolate(tar_img.unsqueeze(0), 
                                          size = (130,130), mode='bilinear').squeeze(0)
    src_img = np.array(src_img*255.0, dtype = np.uint8).transpose(1,2,0)
    tar_img = np.array(low*255.0, dtype = np.uint8).transpose(1,2,0)
    
    scaling_algorithm: SuppScalingAlgorithms = SuppScalingAlgorithms.NEAREST

    scaling_library: SuppScalingLibraries = SuppScalingLibraries.PIL

    scaler_approach: ScalingApproach = ScalingGenerator.create_scaling_approach(x_val_source_shape=src_img.shape,
                                                                                x_val_target_shape=tar_img.shape,
                                                                                lib=scaling_library,
                                                                                alg=scaling_algorithm)

    scale_att: ScaleAttackStrategy = QuadraticScaleAttack(eps=1, verbose=False)

    result_attack_image, deltav, deltah = scale_att.attack(src_image=src_img,
                                                 target_image=tar_img,
                                                 scaler_approach=scaler_approach)
    print('Scaling done')
    lower = scaler_approach.scale_image(xin=result_attack_image)
    
    return result_attack_image, lower, deltav, deltah


def unnorm(X_norm):
    inv_normalize = T.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],std=[1/0.229, 1/0.224, 1/0.225])
    for i in range(X_norm.size(0)):
        X_norm[i] = inv_normalize(X_norm[i])
    return X_norm
