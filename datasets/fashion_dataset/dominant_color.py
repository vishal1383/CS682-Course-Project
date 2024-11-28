import os
from PIL import Image
from sklearn.cluster import KMeans
import numpy as np

#Extracting the dominant color from an image using KMeans clustering
def get_dominant_color(image, n_colors=1):
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    h = image.height // 5 if image.height > 5 else image.height
    w = image.width // 5 if image.width > 5 else image.width
    
    image = image.resize((w, h))  
    image_np = np.array(image)
    pixels = image_np.reshape(-1, 3)
    
    # Using KMeans clustering to find the dominant color
    kmeans = KMeans(n_clusters=n_colors)
    kmeans.fit(pixels)
    
    dominant_color = kmeans.cluster_centers_.astype(int)
    
    return tuple(dominant_color[0])

def save_dominant_color(cropped_image_path, output_folder):
    img = Image.open(cropped_image_path)

    dominant_color = get_dominant_color(img)
    txt_file_path = os.path.join(output_folder, os.path.splitext(os.path.basename(cropped_image_path))[0] + ".txt")
    
    with open(txt_file_path, 'w') as f:
        f.write(f"{dominant_color}")
    
    # print(f"Dominant color for {cropped_image_path} saved as {txt_file_path}")

import os

def process_cropped_images(cropped_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    for img_folder in os.listdir(cropped_folder):
        if img_folder.startswith('.'):
            continue

        for img_file in os.listdir(os.path.join(cropped_folder, img_folder)):
            img_path = os.path.join(cropped_folder, img_folder, img_file)
            
            if os.path.isfile(img_path):
                save_dominant_color(img_path, output_folder)
            else:
                print(f"Skipping {img_path} (not a valid image file).")

    print("Dominant colors processed for all cropped images.")

cropped_folder = './rcnn_cropped_images'
output_folder = './dominant_colours'

process_cropped_images(cropped_folder, output_folder)
