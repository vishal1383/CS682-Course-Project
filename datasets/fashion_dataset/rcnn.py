import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from PIL import Image
import os
import pandas as pd

# Loading the pre-trained Faster R-CNN model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = fasterrcnn_resnet50_fpn(pretrained=True)
model.eval().to(device)

# Folder paths
input_folder = './images'
output_folder = './rcnn_cropped_images'

os.makedirs(output_folder, exist_ok = True)

# Function to detect objects in an image
def detect_objects(image_path, threshold = 0.5):

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
def crop_objects(image, boxes, output_folder, cropped_dest_path):
    cropped_images = []
    # Saving each cropped object with the same image ID (filename)
    for i, box in enumerate(boxes):
        
        x_min, y_min, x_max, y_max = map(int, box)
        
        cropped_image = image.crop((x_min, y_min, x_max, y_max))
        
        # Saving the first crop with the original filename, subsequent ones get the same name
        if i == 0:
            cropped_image.save(cropped_dest_path)
            break
        else:
            # Save crops with less confidence
            # cropped_filename = f"{original_filename.split('.')[0]}_crop_{i}.jpg"
            # cropped_image.save(os.path.join(output_folder, cropped_filename))
            pass

        cropped_images.append(cropped_image)
    
    return cropped_images

def process_images(input_folder, output_folder, K = 5000, overwrite = False):
    data_df = pd.read_csv('dataset.csv')
    image_files = [data_df.iloc[i]['filename'] for i in range(len(data_df))]
    image_files = image_files[:K]

    for idx, image_filename in enumerate(image_files):
        image_path = os.path.join(input_folder, image_filename)

        cropped_folder = os.path.join(output_folder, image_filename.split('.')[0])
        
        if overwrite == False and os.path.exists(cropped_folder):
            continue
        else:
            os.makedirs(cropped_folder, exist_ok = True)

        boxes, image = detect_objects(image_path)
        
        cropped_dest_path = os.path.join(cropped_folder, image_filename)
        crop_objects(image, boxes, output_folder, cropped_dest_path)

        if (idx + 1) % 100 == 0:
            print(f"Processed {idx + 1}/{len(image_files)} images.")

process_images(input_folder, output_folder)
print("Image processing complete.")
