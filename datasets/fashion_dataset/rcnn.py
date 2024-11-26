import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from PIL import Image
import os

# Loading the pre-trained Faster R-CNN model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = fasterrcnn_resnet50_fpn(pretrained=True)
model.eval().to(device)

# Folder paths
input_folder = r'C:\Users\HARSHITHA KOLUKULURU\OneDrive\Desktop\YOLO_fashion\images_downloaded'  
output_folder = r'C:\Users\HARSHITHA KOLUKULURU\OneDrive\Desktop\YOLO_fashion\rcnn_cropped_images'  

os.makedirs(output_folder, exist_ok=True)

# Function to detect objects in an image
def detect_objects(image_path, threshold=0.5):

    image = Image.open(image_path).convert("RGB")
    transform = F.to_tensor
    image_tensor = transform(image).unsqueeze(0).to(device)

    
    with torch.no_grad():
        outputs = model(image_tensor)

    # Extracting boxes for detected objects above threshold
    detections = outputs[0]
    boxes = detections['boxes'][detections['scores'] > threshold]
    return boxes.cpu().numpy(), image

# Function to crop the detected objects from the image
def crop_objects(image, boxes, output_folder, original_filename):
    cropped_images = []
    # Saving each cropped object with the same image ID (filename)
    for i, box in enumerate(boxes):
        
        x_min, y_min, x_max, y_max = map(int, box)
        
        cropped_image = image.crop((x_min, y_min, x_max, y_max))
        
        # Saving the first crop with the original filename, subsequent ones get the same name
        if i == 0:  
            cropped_filename = f"{original_filename}"  
            cropped_image.save(os.path.join(output_folder, cropped_filename))
        else:
            cropped_filename = f"{original_filename.split('.')[0]}_crop_{i}.jpg"
            cropped_image.save(os.path.join(output_folder, cropped_filename))

        cropped_images.append(cropped_image)
    
    return cropped_images

def process_images(input_folder, output_folder):
    image_files = [f for f in os.listdir(input_folder) if f.endswith('.jpg')]

    for image_filename in image_files:
        image_path = os.path.join(input_folder, image_filename)

        boxes, image = detect_objects(image_path)

        crop_objects(image, boxes, output_folder, image_filename)

process_images(input_folder, output_folder)
print("Image processing complete.")
