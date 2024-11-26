import os
from PIL import Image
from sklearn.cluster import KMeans
import numpy as np

#Extracting the dominant color from an image using KMeans clustering
def get_dominant_color(image, n_colors=1):
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    image = image.resize((image.width // 5, image.height // 5))  
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
        f.write(f"Dominant Color (RGB): {dominant_color}\n")
    
    print(f"Dominant color for {cropped_image_path} saved as {txt_file_path}")


import os

def process_cropped_images(cropped_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True) 

    for img_file in os.listdir(cropped_folder):
        img_path = os.path.join(cropped_folder, img_file)
        
        if os.path.isfile(img_path):
            save_dominant_color(img_path, output_folder)
        else:
            print(f"Skipping {img_path} (not a valid image file).")

    print("Dominant colors processed for all cropped images.")

cropped_folder = r"C:\Users\HARSHITHA KOLUKULURU\OneDrive\Desktop\YOLO_fashion\rcnn_cropped_images"  # Path where YOLO saves cropped images
output_folder = r"C:\Users\HARSHITHA KOLUKULURU\OneDrive\Desktop\YOLO_fashion\dominant_colors"  # Folder to save dominant color text files

process_cropped_images(cropped_folder, output_folder)
