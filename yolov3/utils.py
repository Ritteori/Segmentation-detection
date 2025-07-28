import os
import xml
import xml.etree
import xml.etree.ElementTree
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm
import torchvision
from config import ANCHORS

VOC_CLASSES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus",
    "car", "cat", "chair", "cow", "diningtable", "dog",
    "horse", "motorbike", "person", "pottedplant",
    "sheep", "sofa", "train", "tvmonitor"
]

class_to_id = {name: i for i, name in enumerate(VOC_CLASSES)}

def convert_bbox(size:tuple[int,int], bbox:tuple)-> tuple: 
    
    dw = 1 / size[0]
    dh = 1 / size[1]
    
    x_min, y_min, x_max, y_max = bbox
    
    x_center = (x_max + x_min) / 2.0
    y_center = (y_max + y_min) / 2.0
    
    width = x_max - x_min
    height = y_max - y_min
    
    return x_center * dw, y_center * dh, width * dw, height * dh

def convert_xml_to_bboxes(xml_file:str, labels_dir:str):
    
    tree = xml.etree.ElementTree.parse(xml_file)
    root = tree.getroot()
    
    size = root.find('size')
    width = int(size.find('width').text)
    height = int(size.find('height').text)
    
    yolo_lines = []
    
    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in class_to_id or int(difficult) == 1:
            continue
        
        cls_id = class_to_id[cls]
        
        xmlbox = obj.find('bndbox')
        bbox = (
            float(xmlbox.find('xmin').text),
            float(xmlbox.find('ymin').text),
            float(xmlbox.find('xmax').text),
            float(xmlbox.find('ymax').text)
        )
        
        converted_bbox = convert_bbox((width,height),bbox)
        
        yolo_line = f'{cls_id} ' + ' '.join(f"{x:.6f}" for x in converted_bbox)
        yolo_lines.append(yolo_line)
        
    base_filename =  os.path.splitext(os.path.basename(xml_file))[0]
    text_path = os.path.join(labels_dir,base_filename + '.txt')
    with open(text_path,'a') as f:
        f.write("\n".join(yolo_lines))
        
def convert_all_xmls_to_yolo(annotations_dir:str,labels_dir:str):
    
    os.makedirs(labels_dir,exist_ok=True)
    xml_files = [path for path in os.listdir(annotations_dir) if path.endswith('.xml')]
    print(f"Found {len(xml_files)} xml files.")
    
    for xml_file in xml_files:
        full_path = os.path.join(annotations_dir,xml_file)
        convert_xml_to_bboxes(full_path,labels_dir)
        
    print("Convertation is ready")


def yolo_collate_fn(batch):
    images, boxes, classes = zip(*batch)

    images = torch.stack(images)

    targets = []
    for b, c in zip(boxes, classes):
        b = torch.tensor(b, dtype=torch.float32)
        c = torch.tensor(c, dtype=torch.float32).unsqueeze(1)
        target = torch.cat([b, c], dim=1)  # [N,5]
        targets.append(target)

    return images, targets


def get_best_anchor_idxs(gt_boxes_wh, anchors_wh):

    # gt_boxes_wh : [N,2], anchors_wh : [A,2] , где 2 - (width,height)

    gt_boxes_wh = gt_boxes_wh.unsqueeze(1)
    anchors_wh = anchors_wh.unsqueeze(0)
    
    inter_wh = torch.min(gt_boxes_wh, anchors_wh)
    inter_area = inter_wh[..., 0] * inter_wh[..., 1]
    
    area_gt = gt_boxes_wh[...,0] *  gt_boxes_wh[...,1]
    area_anchor = anchors_wh[...,0] * anchors_wh[...,1]
    unioun_area = area_gt + area_anchor - inter_area
    
    iou = inter_area / (unioun_area + 1e-6)
    best_anchor = torch.argmax(iou,dim=1)
    
    return best_anchor, iou

def train_loop(model, dataloader, optimizer, loss_fn, device, epochs, save_path=None):
    model.to(device)

    scaler = torch.amp.GradScaler('cuda') 
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        total_batches = 0

        for images, targets in tqdm(dataloader):
            images = images.to(device)
            targets = [t.to(device) for t in targets]

            optimizer.zero_grad()
            
            with torch.amp.autocast('cuda'):
                preds = model(images)
                loss = loss_fn(preds, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            
            scaler.update()

            running_loss += loss.item()
            total_batches += 1

        avg_loss = running_loss / total_batches
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        
        show_results(model, dataloader, device, anchors=ANCHORS, epoch=epoch, num_images=9, conf_threshold=0.3)

        if save_path is not None:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, save_path)


def decode_predictions(preds, anchors, num_classes=20, conf_threshold=0.5, iou_threshold=0.4, img_size=416, device='cpu'):
    """
    preds: tuple из 3 тензоров с выходами модели для разных scale,
           каждый [B, A*(5+C), S, S]
    anchors: list из 3 списков с якорями для каждого scale, например:
             [[(10,13),(16,30),(33,23)], [(30,61),(62,45),(59,119)], [(116,90),(156,198),(373,326)]]
    Возвращает: список предсказанных боксов для батча.
    Каждый бокс: [x1, y1, x2, y2, confidence, class_id]
    """

    model_boxes = []

    batch_size = preds[0].size(0)

    for b in range(batch_size):
        boxes = []

        for scale_idx, pred in enumerate(preds):
            anchors_scale = torch.tensor(anchors[scale_idx], dtype=torch.float32, device=device)
            S = pred.size(2)
            num_anchors = anchors_scale.size(0)

            # reshape и перестановка
            pred = pred[b].view(num_anchors, 5 + num_classes, S, S)
            pred = pred.permute(0, 2, 3, 1)  # [A, S, S, 5+C]

            # сигмоид для xy и obj
            x = torch.sigmoid(pred[..., 0])
            y = torch.sigmoid(pred[..., 1])
            w = pred[..., 2]
            h = pred[..., 3]
            obj = torch.sigmoid(pred[..., 4])
            class_scores = torch.softmax(pred[..., 5:], dim=-1)

            # grid
            grid_y, grid_x = torch.meshgrid(torch.arange(S, device=device), torch.arange(S, device=device), indexing='ij')
            grid_x = grid_x.unsqueeze(0)
            grid_y = grid_y.unsqueeze(0)

            # box координаты (центр)
            bx = (x + grid_x) / S
            by = (y + grid_y) / S

            # размеры (w,h)
            bw = torch.exp(w) * anchors_scale[:, 0].view(-1,1,1) / img_size
            bh = torch.exp(h) * anchors_scale[:, 1].view(-1,1,1) / img_size

            # confidence на объект
            conf = obj

            # max класс + score
            class_prob, class_id = torch.max(class_scores, dim=-1)

            # итоговый confidence с классом
            conf_mask = conf * class_prob > conf_threshold

            if conf_mask.sum() == 0:
                continue

            bx = bx[conf_mask]
            by = by[conf_mask]
            bw = bw[conf_mask]
            bh = bh[conf_mask]
            conf = conf[conf_mask]
            class_id = class_id[conf_mask]

            # пересчёт из центра + w,h в x1,y1,x2,y2 (относительно 0..1)
            x1 = bx - bw / 2
            y1 = by - bh / 2
            x2 = bx + bw / 2
            y2 = by + bh / 2

            # собираем боксы
            for i in range(x1.size(0)):
                boxes.append([x1[i].item(), y1[i].item(), x2[i].item(), y2[i].item(), conf[i].item(), class_id[i].item()])

        # Применяем NMS на все боксы батча (переведём в тензор)
        if len(boxes) == 0:
            model_boxes.append([])
            continue

        boxes_tensor = torch.tensor(boxes, device=device)
        # boxes_tensor: [N,6]: x1,y1,x2,y2,conf,class_id

        # torchvision NMS принимает [x1,y1,x2,y2] и scores
        keep = torchvision.ops.nms(boxes_tensor[:, :4], boxes_tensor[:,4], iou_threshold)

        filtered_boxes = boxes_tensor[keep].cpu().numpy().tolist()
        model_boxes.append(filtered_boxes)

    return model_boxes

import numpy as np

def show_results(model, dataloader, device, anchors, epoch, num_images=9, conf_threshold=0.5):
    model.eval()
    images_shown = 0

    fig, axs = plt.subplots(3, 3, figsize=(12, 12))
    axs = axs.flatten()

    if epoch > 10:
        conf_threshold = 0.5
    
    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            preds = model(images)

            batch_size = images.size(0)
            boxes_batch = decode_predictions(preds, anchors=anchors, device=device, conf_threshold=conf_threshold)

            for i in range(min(batch_size, num_images - images_shown)):
                img = images[i].cpu().permute(1, 2, 0).numpy()
                img = np.clip(img, 0, 1)
                img_h, img_w = img.shape[:2]

                boxes = boxes_batch[i]

                ax = axs[images_shown]
                ax.imshow(img)
                ax.axis('off')

                for box in boxes:
                    x1, y1, x2, y2, conf, class_id = box

                    # Преобразуем координаты из относительных (0..1) в пиксели
                    x1_pix = x1 * img_w
                    y1_pix = y1 * img_h
                    x2_pix = x2 * img_w
                    y2_pix = y2 * img_h

                    width = x2_pix - x1_pix
                    height = y2_pix - y1_pix

                    rect = plt.Rectangle((x1_pix, y1_pix), width, height, edgecolor='r', facecolor='none', linewidth=2)
                    ax.add_patch(rect)

                    class_name = VOC_CLASSES[int(class_id)]
                    ax.text(x1_pix, y1_pix - 5, f"{class_name}: {conf:.2f}", color='yellow', fontsize=8, backgroundcolor='black')

                images_shown += 1
                if images_shown >= num_images:
                    break

            if images_shown >= num_images:
                break

    plt.tight_layout()
    plt.savefig(fr'J:\ML_for_porfolio\segmentation\yolov3\results\epoch_{epoch + 1}.png')
    model.train()


            
            
        