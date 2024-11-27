import torch
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPModel, CLIPProcessor, AdamW
from PIL import Image
import os
import pandas as pd
import torch.nn.functional as F

class CustomCLIPDataset(Dataset):
    def __init__(self, images, texts, processor, max_length=77):
        self.images = images
        self.texts = texts
        self.processor = processor
        self.max_length = max_length
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert("RGB")
        processed_image = self.processor(images=image, return_tensors="pt")['pixel_values'].squeeze()
        
        processed_text = self.processor(text=self.texts[idx], 
                                     return_tensors="pt",  
                                     truncation=True,
                                     padding='max_length',
                                     max_length=self.max_length)
        
        return {
            'image': processed_image,
            'text': processed_text['input_ids'].squeeze(),
            'attention_mask': processed_text['attention_mask'].squeeze()
        }

def info_nce_loss(image_features, text_features, temperature=0.07):
    batch_size = image_features.shape[0]
    
    # Normalize features
    image_features = F.normalize(image_features, dim=-1)
    text_features = F.normalize(text_features, dim=-1)
    
    # Calculate similarity matrix
    logits = torch.matmul(image_features, text_features.T) / temperature
    
    # Labels are all diagonal elements (where image and text match)
    labels = torch.arange(batch_size, device=image_features.device)
    
    # Calculate loss for both image->text and text->image direction
    loss_i2t = F.cross_entropy(logits, labels)
    loss_t2i = F.cross_entropy(logits.T, labels)
    
    # Total loss is the average of both directions
    total_loss = (loss_i2t + loss_t2i) / 2
    return total_loss

def fine_tune_clip(images, texts, model_name='openai/clip-vit-base-patch32', 
                   learning_rate=5e-5, batch_size=16, epochs=30):
    model = CLIPModel.from_pretrained(model_name)
    processor = CLIPProcessor.from_pretrained(model_name)
    
    dataset = CustomCLIPDataset(images, texts, processor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            images = batch['image'].to(device)
            texts = batch['text'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            optimizer.zero_grad()
            
            # Get image and text features
            outputs = model(input_ids=texts, 
                          attention_mask=attention_mask, 
                          pixel_values=images,
                          return_loss=False)
            
            # Calculate InfoNCE loss
            loss = info_nce_loss(outputs.image_embeds, outputs.text_embeds)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader)}")
    
    model = model.to('cpu')
    model.save_pretrained("./CLIPcheckpoints/fine_tuned_clip_model", safe_serialization=True)
    processor.save_pretrained("./CLIPcheckpoints/fine_tuned_clip_processor")
    
    return model, processor

def load_checkpoints():
    model = CLIPModel.from_pretrained("./CLIPcheckpoints/fine_tuned_clip_model")
    processor = CLIPProcessor.from_pretrained("./CLIPcheckpoints/fine_tuned_clip_processor")
    return model, processor

if __name__ == "__main__":
    prefix = "../../datasets/fashion_dataset/rcnn_cropped_images/"
    images = set([file for file in os.listdir(prefix) if "crop" in file])
    dataset = pd.read_csv("../../datasets/fashion_dataset/dataset.csv")
    
    pairs = [(dataset.iloc[i]['query'], prefix + str(dataset.iloc[i]['id'])+"_crop_1.jpg") 
             for i in range(len(dataset)) if str(dataset.iloc[i]['id'])+"_crop_1.jpg" in images]
    
    image_paths = [i[1] for i in pairs]
    texts = [i[0] for i in pairs]
    
    fine_tuned_model, fine_tuned_processor = fine_tune_clip(image_paths, texts)
    load_checkpoints()
    print("Checkpoints loaded!")