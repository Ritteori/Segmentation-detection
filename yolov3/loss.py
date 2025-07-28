import torch
from torch import nn

from utils import get_best_anchor_idxs

class YOLOLoss(nn.Module):
    def __init__(self, anchors, num_classes=20, img_shape=(3,416,416), ignore_iou_thresh=0.5):
        super().__init__()
        self.num_classes = num_classes
        self.img_shape = img_shape
        self.ignore_iou_thresh = ignore_iou_thresh
        self.anchors = [torch.tensor(scale, dtype=torch.float32) for scale in anchors]

        self.mse = nn.MSELoss(reduction='sum')
        self.bce = nn.BCEWithLogitsLoss(reduction='sum')
        self.ce = nn.CrossEntropyLoss(reduction='sum')

    def forward(self, preds, targets):
        total_loss = 0
        batch_size = preds[0].size(0)

        device = preds[0].device

        for scale_idx, pred in enumerate(preds):
            anchors = self.anchors[scale_idx].to(device)
            S = pred.size(2)
            num_anchors = anchors.size(0)

            pred = pred.view(batch_size, num_anchors, 5 + self.num_classes, S, S)
            pred = pred.permute(0, 1, 3, 4, 2).contiguous()  # B x A x S x S x (5+C)

            x = torch.sigmoid(pred[..., 0]) 
            y = torch.sigmoid(pred[..., 1])  
            w = pred[..., 2]                 
            h = pred[..., 3]                 
            obj = pred[..., 4]               
            class_logits = pred[..., 5:]     

            grid_y, grid_x = torch.meshgrid(torch.arange(S), torch.arange(S), indexing='ij')
            grid_x = grid_x.to(device).float()
            grid_y = grid_y.to(device).float()

            pred_boxes = torch.zeros_like(pred[..., :4])
            pred_boxes[..., 0] = (x + grid_x) / S
            pred_boxes[..., 1] = (y + grid_y) / S
            pred_boxes[..., 2] = torch.exp(w) * anchors[:, 0].view(1, num_anchors, 1, 1) / self.img_shape[2]
            pred_boxes[..., 3] = torch.exp(h) * anchors[:, 1].view(1, num_anchors, 1, 1) / self.img_shape[1]

            obj_target = torch.zeros_like(obj)
            x_target = torch.zeros_like(x)
            y_target = torch.zeros_like(y)
            w_target = torch.zeros_like(w)
            h_target = torch.zeros_like(h)
            class_target = torch.zeros(batch_size, num_anchors, S, S, dtype=torch.long, device=device)

            for b in range(batch_size):
                target = targets[b].to(device)  # [num_boxes, 5]: x,y,w,h,class
                if target.numel() == 0:
                    continue

                gt_wh = target[:, 2:4]

                anchors_wh = anchors / torch.tensor([self.img_shape[2], self.img_shape[1]], device=device)  # [num_anchors, 2]

                best_anchor_idxs, _ = get_best_anchor_idxs(gt_wh, anchors_wh)  # [num_boxes]

                for i, t in enumerate(target):
                    gx, gy, gw, gh, cls = t[0], t[1], t[2], t[3], int(t[4])
                    best_anchor = best_anchor_idxs[i]

                    gx_cell = gx * S
                    gy_cell = gy * S
                    cell_x = int(gx_cell)
                    cell_y = int(gy_cell)

                    obj_target[b, best_anchor, cell_y, cell_x] = 1
                    x_target[b, best_anchor, cell_y, cell_x] = gx_cell - cell_x
                    y_target[b, best_anchor, cell_y, cell_x] = gy_cell - cell_y
                    w_target[b, best_anchor, cell_y, cell_x] = torch.log(gw / anchors[best_anchor][0] * self.img_shape[2] + 1e-16)
                    h_target[b, best_anchor, cell_y, cell_x] = torch.log(gh / anchors[best_anchor][1] * self.img_shape[1] + 1e-16)
                    class_target[b, best_anchor, cell_y, cell_x] = cls

            loss_obj = self.bce(obj, obj_target)

            loss_x = self.mse(x * obj_target, x_target * obj_target)
            loss_y = self.mse(y * obj_target, y_target * obj_target)
            loss_w = self.mse(w * obj_target, w_target * obj_target)
            loss_h = self.mse(h * obj_target, h_target * obj_target)

            pred_cls_flat = class_logits[obj_target.bool()]  # [N, num_classes]
            target_cls_flat = class_target[obj_target.bool()].to(torch.long)  # [N]

            if pred_cls_flat.numel() > 0:
                loss_cls = self.ce(pred_cls_flat, target_cls_flat)
            else:
                loss_cls = torch.tensor(0., device=device)

            total_loss += loss_obj + loss_x + loss_y + loss_w + loss_h + loss_cls

        return total_loss / batch_size