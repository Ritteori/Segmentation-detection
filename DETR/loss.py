import torch
from torch import nn
from torch.nn import functional as F
from scipy.optimize import linear_sum_assignment

class DETRLoss(nn.Module):
    def __init__(self, alpha=1, beta=1, gamma=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
    
    def cost_class(self, pred_labels, target_labels):
        
        bs = pred_labels.size(0)
        
        costs = []
        
        for i in range(bs):
            
            logits_labels_i = pred_labels[i] # (num_queries,num_classes)
            
            pred_labels_i = torch.argmax(logits_labels_i,dim=-1) # (num_queries)
            target_labels_i = target_labels[i] # (num_objects)
            
            pred_labels_i = pred_labels_i.unsqueeze(-1) # (num_queries,1)
            target_labels_i = target_labels_i.unsqueeze(0) # (1,num_objects)
            
            pred_labels_i = pred_labels_i.to(torch.float32)
            cost_mattrix = pred_labels_i @ target_labels_i # (num_queries,num_objects)
            costs.append(cost_mattrix)

        return costs # (bs,num_queries,num_objects)          
    
    def cost_bboxes(self, pred_boxes, target_boxes):
        
        bs = pred_boxes.size(0)
        
        bboxes_costs = []
        ious = []
        
        for i in range(bs):
            
            pred_boxes_i = pred_boxes[i] # (num_queries,4)
            target_boxes_i = target_boxes[i] # (num_objects,4)
            
            pred_boxes_i = pred_boxes_i.unsqueeze(1) # (num_queries,1,4)
            target_boxes_i = target_boxes_i.unsqueeze(0) # (1,num_objects,4)
            
            bboxes_cost = ((pred_boxes_i - target_boxes_i) ** 2).sum(-1) # (num_queries,num_objects)
            iou_cost = self.iou(pred_boxes_i,target_boxes_i)
            
            ious.append(iou_cost)
            bboxes_costs.append(bboxes_cost)
            
        return bboxes_costs, ious # (bs,num_queries,num_objects)
    
    def iou(self,pred_boxes, target_boxes):
        
        pred_x1 = pred_boxes[...,0] - pred_boxes[...,2] / 2
        pred_y1 = pred_boxes[...,1] - pred_boxes[...,3] / 2
        pred_x2 = pred_boxes[...,0] + pred_boxes[...,2] / 2
        pred_y2 = pred_boxes[...,1] + pred_boxes[...,3] / 2
        
        target_x1 = target_boxes[...,0] - target_boxes[...,2] / 2
        target_y1 = target_boxes[...,1] - target_boxes[...,3] / 2
        target_x2 = target_boxes[...,0] - target_boxes[...,2] / 2
        target_y2 = target_boxes[...,1] - target_boxes[...,3] / 2
        
        inter_x1 = torch.max(pred_x1,target_x1)
        inter_y1 = torch.max(pred_y1,target_y1)
        inter_x2 = torch.min(pred_x2,target_x2)
        inter_y2 = torch.min(pred_y2,target_y2)
        
        inter_w = inter_x2 - inter_x1
        inter_h = inter_y2 - inter_y1
        
        inter_area = inter_h * inter_w
        
        pred_area = pred_boxes[...,2] * pred_boxes[...,3]
        target_area = target_boxes[...,2] * target_boxes[...,3]
        
        union_area = pred_area + target_area - inter_area
        
        iou = inter_area / (union_area + 1e-6)
        
        return iou # (num_queries,num_objects)
    
    def forward(self,pred_labels,pred_boxes,target_labels,target_boxes):
        
        # pred_labels: (bs,num_queries,num_classes)
        # pred_boxes: (bs,num_queries,4)
        # target_labels: (bs,num_objects)
        # target_boxes: (bs,num_objects,4)
        
        labels_costs = self.cost_class(pred_labels,target_labels)
        bboxes_costs, iou_costs = self.cost_bboxes(pred_boxes,target_boxes)
        
        labels_costs = torch.stack(labels_costs, dim=0)
        bboxes_costs = torch.stack(bboxes_costs, dim=0)
        iou_costs    = torch.stack(iou_costs, dim=0)
        
        total_cost = self.alpha * labels_costs + self.beta * bboxes_costs - self.gamma * iou_costs # (bs,num_queries,num_objects)
        
        bs = total_cost.size(0)
        
        indices = []
        for i in range(bs):
            total_cost_i = total_cost[i].detach().cpu().numpy()
            
            queries_indices, object_indices = linear_sum_assignment(total_cost_i)
            
            indices.append(
                (
                    torch.tensor(queries_indices,dtype=torch.float32),
                    torch.tensor(object_indices,dtype=torch.float32),
                )
            )
        
        return indices # (bs,2,num_objects)
            
            