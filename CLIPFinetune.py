import torch
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPModel, CLIPProcessor, AdamW
from PIL import Image

class CustomCLIPDataset(Dataset):
    def __init__(self, images, texts, processor):
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
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # Load and preprocess image
        image = Image.open(self.images[idx]).convert("RGB")
        processed_image = self.processor(images=image, return_tensors="pt")['pixel_values'].squeeze()
        
        # Preprocess text
        processed_text = self.processor(text=self.texts[idx], return_tensors="pt", padding=True, truncation=True)
        
        return {
            'image': processed_image,
            'text': processed_text['input_ids'].squeeze(),
            'attention_mask': processed_text['attention_mask'].squeeze()
        }

def fine_tune_clip(images, texts, model_name='openai/clip-vit-base-patch32', 
                   learning_rate=5e-5, batch_size=16, epochs=3):
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
    
    # Prepare dataset and dataloader
    dataset = CustomCLIPDataset(images, texts, processor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
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
    
    # Save fine-tuned model
    model.save_pretrained("./fine_tuned_clip_model")
    processor.save_pretrained("./fine_tuned_clip_processor")
    
    return model, processor

# Example usage
if __name__ == "__main__":
    # Replace with your actual image paths and corresponding text descriptions
    image_paths = [
        "path/to/image1.jpg", 
        "path/to/image2.jpg", 
        # Add more image paths
    ]
    
    texts = [
        "Description of image 1", 
        "Description of image 2", 
        # Add corresponding texts
    ]
    
    fine_tuned_model, fine_tuned_processor = fine_tune_clip(image_paths, texts)