from torch.utils.data import Dataset
import os
import cv2
import numpy as np

class YoloDataset(Dataset):
    def __init__(self, labels_path, images_path, transforms=None):
        super().__init__()
        
        self.labels_path = labels_path
        self.images_path = images_path
        self.transforms = transforms
    
        images_list = os.listdir(images_path)
        self.images_list = [os.path.join(self.images_path, image) for image in images_list]
        self.jpeg_names = [os.path.basename(p) for p in self.images_list]
        
    def __len__(self):
        return len(self.images_list)
    
    def __getitem__(self, index):
        
        image = cv2.imread(self.images_list[index])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        
        img_width, img_height = image.shape[1], image.shape[0]
        
        label_path = os.path.join(self.labels_path,self.jpeg_names[index].replace('jpg', 'txt'))
        
        bboxes = []
        classes = []
        
        with open(label_path, "r") as f:
            
            for line in f:
                cls_id, x_center, y_center, width, height = map(float,line.strip().split())
                classes.append(int(cls_id))
                bboxes.append([x_center,y_center,width,height])
    
        bboxes = np.array(bboxes,dtype=np.float32)
        classes = np.array(classes,dtype=np.int64)
        
        if self.transforms is not None:
            
            bboxes_absolute = []
            
            for box in bboxes:
                
                x_center,y_center,width,height = box

                x_center *= img_width
                y_center *= img_height
                width *= img_width
                height *= img_height
                
                x_min = x_center - width / 2.0
                y_min = y_center - height / 2.0
                x_max = x_center + width / 2.0
                y_max = y_center + height / 2.0
                
                bboxes_absolute.append([x_min,y_min,x_max,y_max])
                
            bboxes_absolute = np.array(bboxes_absolute,dtype=np.float32)
            
            transformed = self.transforms(
                image = image,
                bboxes = bboxes_absolute,
                class_labels = classes
            )
            
            image = transformed["image"]
            bboxes_absolute = transformed["bboxes"]
            classes = transformed["class_labels"]
            
            boxes = []
            
            for box in bboxes_absolute:
                
                x_min,y_min,x_max,y_max = box
                
                x_center = (x_max + x_min) / width
                y_center = (y_max + y_min) / height
                width = (x_max - x_min) / width
                height = (y_max - y_min) / height
                boxes.append([x_center, y_center, width, height])
            
            boxes = np.array(boxes, dtype=np.float32)
        else:
            boxes = bboxes
            
        return image, boxes, classes