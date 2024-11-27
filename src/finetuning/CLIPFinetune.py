import torch
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPModel, CLIPProcessor, AdamW
from PIL import Image
import os
import pandas as pd
import os
os.environ['CUDA_HOME'] = os.getenv('CONDA_PREFIX')



class CustomCLIPDataset(Dataset):
    def __init__(self, images, texts, processor, max_length=77):
        """
        Custom dataset for CLIP fine-tuning
        
        Args:
            images (list): List of image file paths or PIL Image objects
            texts (list): Corresponding list of text descriptions
            processor (CLIPProcessor): CLIP processor for image and text preprocessing
        """
        self.images = images
        self.texts = texts
        self.processor = processor
        self.max_length = max_length

    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # Load and preprocess image
        image = Image.open(self.images[idx]).convert("RGB")
        processed_image = self.processor(images=image, return_tensors="pt")['pixel_values'].squeeze()
        
        # Preprocess text
        processed_text = self.processor(text=self.texts[idx], 
                                        return_tensors="pt",  
                                        truncation=True,
                                        padding='max_length',
                                        max_length=self.max_length,)
        
        return {
            'image': processed_image,
            'text': processed_text['input_ids'].squeeze(),
            'attention_mask': processed_text['attention_mask'].squeeze()
        }
    


def fine_tune_clip(images, texts, model_name='openai/clip-vit-base-patch32', 
                   learning_rate=5e-5, batch_size=16, epochs=30):
    """
    Fine-tune CLIP model on custom image-text pairs
    
    Args:
        images (list): List of image file paths
        texts (list): Corresponding list of text descriptions
        model_name (str): Pre-trained CLIP model to start from
        learning_rate (float): Learning rate for optimization
        batch_size (int): Training batch size
        epochs (int): Number of training epochs
    """
    # Load pre-trained CLIP model and processor
    model = CLIPModel.from_pretrained(model_name)
    processor = CLIPProcessor.from_pretrained(model_name)
    
    # def collate_fn(batch):
    #     # Get max length in this batch
    #     max_text_length = max(len(item['text']) for item in batch)
        
    #     # Pad text and attention masks
    #     for item in batch:
    #         padding_length = max_text_length - len(item['text'])
    #         item['text'] = item['text'] + [processor.tokenizer.pad_token_id] * padding_length
    #         item['attention_mask'] = item['attention_mask'] + [0] * padding_length
        
    #     return {
    #         'image': torch.stack([item['image'] for item in batch]),
    #         'text': torch.tensor([item['text'] for item in batch]),
    #         'attention_mask': torch.tensor([item['attention_mask'] for item in batch])
    #     }
    # Prepare dataset and dataloader
    dataset = CustomCLIPDataset(images, texts, processor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    # Prepare optimizer and device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    
    # Training loop
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            # Move batch to device
            images = batch['image'].to(device)
            texts = batch['text'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(input_ids=texts, 
                            attention_mask=attention_mask, 
                            pixel_values=images,
                            return_loss=True)
            loss = outputs.loss
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader)}")
    
    model = model.to('cpu')  # Move to CPU first if needed
    model.save_pretrained("./CLIPcheckpoints/fine_tuned_clip_model", safe_serialization=True)
    processor.save_pretrained("./CLIPcheckpoints/fine_tuned_clip_processor")
    
    return model, processor

def load_checkpoints():
    model = CLIPModel.from_pretrained("./CLIPcheckpoints/fine_tuned_clip_model")
    processor = CLIPProcessor.from_pretrained("./CLIPcheckpoints/fine_tuned_clip_processor")
# Example usage
if __name__ == "__main__":
    # Replace with your actual image paths and corresponding text descriptions
    prefix = "../../datasets/fashion_dataset/rcnn_cropped_images/"
    images = set([file for file in os.listdir(prefix) if "crop" in file])
    dataset = pd.read_csv("../../datasets/fashion_dataset/dataset.csv")
    # image_paths = [ for i in range(len(dataset))]
    # texts = [ for i in range(len(dataset))]
    pairs = [(dataset.iloc[i]['query'],prefix + str(dataset.iloc[i]['id'])+"_crop_1.jpg") 
             for i in range(len(dataset)) if str(dataset.iloc[i]['id'])+"_crop_1.jpg" in images]
    print(len(pairs))
    print(pairs)
    image_paths = [i[1] for i in pairs]
    texts = [i[0] for i in pairs]
    
    fine_tuned_model, fine_tuned_processor = fine_tune_clip(image_paths, texts)

    # Test if you are able to load the checkpoints
    load_checkpoints()
    print("Checkpoints loaded!")
