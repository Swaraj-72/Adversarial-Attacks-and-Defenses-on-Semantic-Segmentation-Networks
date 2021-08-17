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
    
    return np.mean(present_iou_list)*100, acc*100


def pgd(model, X, y, epsilon, alpha, num_iter): # Untargetted Attack 1
    
    delta = torch.zeros_like(X, requires_grad=True)
    trg = y.squeeze(1)
    
    for t in range(num_iter):
        loss = nn.CrossEntropyLoss(ignore_index = 255)(model(X + delta)['out'], trg.long())
        loss.backward()
        print('Loss after iteration {}: {:.2f}'.format(t+1, loss.item()))
        delta.data = (delta + X.shape[0]*alpha*delta.grad.data).clamp(-epsilon,epsilon)
        delta.grad.zero_()
        
    return delta.detach()

def pgd_steep(model, X, y, epsilon, alpha, num_iter): # Untargetted Attack 2
    
    delta = torch.zeros_like(X, requires_grad=True)
    trg = y.squeeze(1)
    
    for t in range(num_iter):
        loss = nn.CrossEntropyLoss(ignore_index = 255)(model(X + delta)['out'], trg.long())
        loss.backward()
        print('Loss after iteration {}: {:.2f}'.format(t+1, loss.item()))
        delta.data = (delta + alpha*delta.grad.detach().sign()).clamp(-epsilon,epsilon)
        delta.grad.zero_()
        
    return delta.detach()
