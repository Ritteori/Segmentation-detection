import os
import xml
import xml.etree
import xml.etree.ElementTree


VOC_CLASSES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus",
    "car", "cat", "chair", "cow", "diningtable", "dog",
    "horse", "motorbike", "person", "pottedplant",
    "sheep", "sofa", "train", "tvmonitor"
]

class_to_id = {VOC_CLASSES[i - 1]:i for i in range(1,len(VOC_CLASSES) + 1)}

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
        
        yolo_line = f'{cls_id} ' + ''.join(f"{x:.6f}" for x in converted_bbox)
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
        