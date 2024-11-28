import torch
from torch.utils.data import DataLoader, random_split
from transformers import CLIPModel, CLIPProcessor, AdamW
import os
import torch.nn.functional as F

from src.finetuning.custom_clip_dataset import CustomCLIPDataset


dataset_paths = {
    'deep_fashion': 'fashion_dataset',
    'test_data': 'dataset'
}
    
class CLIPFinetune():

    def __init__(self, dataset_type, images, texts, model_name='openai/clip-vit-base-patch32', learning_rate=5e-5, batch_size=16, epochs=30, patience=1, val_split=0.1):
        self.images = images

         # Load pre-trained CLIP model and processor
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)

        # Prepare dataset and dataloader
        dataset = CustomCLIPDataset(images, texts, self.processor)
        
        val_size = int(len(dataset) * val_split)
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size)

        # Prepare optimizer and device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        self.patience = patience

        self.epochs = epochs
        
        self.dataset_type = dataset_type
        self.models_path = os.path.join('../models', dataset_paths[self.dataset_type], 'finetune')
        os.makedirs(self.models_path, exist_ok = True)

    def info_nce_loss(self, image_features, text_features, temperature=0.8):
        batch_size = image_features.shape[0]
        
        # Normalize features
        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)
        
        # Calculate similarity matrix
        # logits = torch.matmul(image_features, text_features.T) / temperature
        logits = torch.matmul(text_features, image_features.T) / temperature
        # Labels are all diagonal elements
        labels = torch.arange(batch_size, device=image_features.device)
        
        # Calculate loss for both directions
        loss_t2i = F.cross_entropy(logits, labels)
        # loss_t2i = F.cross_entropy(logits.T, labels)
        return loss_t2i
        # return (loss_i2t + loss_t2i) / 2


    def train(self):
        best_val_loss = float('inf')
        patience_counter = 0
        best_model = None
        
        for epoch in range(self.epochs):
            # Training phase
            self.model.train()
            train_loss = 0
            for batch in self.train_loader:
                images = batch['image'].to(self.device)
                texts = batch['text'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(input_ids=texts, 
                            attention_mask=attention_mask, 
                            pixel_values=images,
                            return_loss=False)
                
                loss = self.info_nce_loss(outputs.image_embeds, outputs.text_embeds)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
            
            # Validation phase
            val_loss = self.evaluate(self.model, self.val_loader, self.device)
            
            print(f"Epoch {epoch+1}/{self.epochs}")
            print(f"Training Loss: {train_loss/len(self.train_loader):.4f}")
            print(f"Validation Loss: {val_loss:.4f}")
            
            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model = self.model.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    print(f"Early stopping triggered after {epoch+1} epochs")
                    break
         
        self.model.load_state_dict(best_model)
        self.model = self.model.to('cpu')

        self.model.save_pretrained(os.path.join(self.models_path, "fine_tuned_clip_modelvlm"), safe_serialization=True)
        self.processor.save_pretrained(os.path.join(self.models_path, "fine_tuned_clip_processorvlm"))
        return self.model, self.processor

    def evaluate(self, dataloader):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch in dataloader:
                images = batch['image'].to(self.device)
                texts = batch['text'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                outputs = self.model(input_ids=texts, 
                            attention_mask=attention_mask, 
                            pixel_values=images,
                            return_loss=False)
                
                loss = self.info_nce_loss(outputs.image_embeds, outputs.text_embeds)
                total_loss += loss.item()
        
        return total_loss / len(dataloader)     


# def fine_tune_clip(self, images, texts, model_name='openai/clip-vit-base-patch32', 
#                    learning_rate=5e-5, batch_size=16, epochs=30):
#     """
#     Fine-tune CLIP model on custom image-text pairs
    
#     Args:
#         images (list): List of image file paths
#         texts (list): Corresponding list of text descriptions
#         model_name (str): Pre-trained CLIP model to start from
#         learning_rate (float): Learning rate for optimization
#         batch_size (int): Training batch size
#         epochs (int): Number of training epochs
#     """
#     # Load pre-trained CLIP model and processor
#     model = CLIPModel.from_pretrained(model_name)
#     processor = CLIPProcessor.from_pretrained(model_name)
    
#     # def collate_fn(batch):
#     #     # Get max length in this batch
#     #     max_text_length = max(len(item['text']) for item in batch)
        
#     #     # Pad text and attention masks
#     #     for item in batch:
#     #         padding_length = max_text_length - len(item['text'])
#     #         item['text'] = item['text'] + [processor.tokenizer.pad_token_id] * padding_length
#     #         item['attention_mask'] = item['attention_mask'] + [0] * padding_length
        
#     #     return {
#     #         'image': torch.stack([item['image'] for item in batch]),
#     #         'text': torch.tensor([item['text'] for item in batch]),
#     #         'attention_mask': torch.tensor([item['attention_mask'] for item in batch])
#     #     }
#     # Prepare dataset and dataloader
#     dataset = CustomCLIPDataset(images, texts, processor)
#     dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
#     # Prepare optimizer and device
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model.to(device)
#     optimizer = AdamW(model.parameters(), lr=learning_rate)
    
#     # Training loop
#     model.train()
#     for epoch in range(epochs):
#         total_loss = 0
#         for batch in dataloader:
#             # Move batch to device
#             images = batch['image'].to(device)
#             texts = batch['text'].to(device)
#             attention_mask = batch['attention_mask'].to(device)
            
#             # Zero gradients
#             optimizer.zero_grad()
            
#             # Forward pass
#             outputs = model(input_ids=texts, 
#                             attention_mask=attention_mask, 
#                             pixel_values=images,
#                             return_loss=True)
#             loss = outputs.loss
            
#             # Backward pass and optimization
#             loss.backward()
#             optimizer.step()
            
#             total_loss += loss.item()
        
#         print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader)}")
    
#     model = model.to('cpu')  # Move to CPU first if needed
#     model.save_pretrained(os.path.join(self.models_path, "fine_tuned_clip_model"), safe_serialization=True)
#     processor.save_pretrained("./CLIPcheckpoints/fine_tuned_clip_processor")
    
#     return model, processor

# def load_checkpoints():
#     model = CLIPModel.from_pretrained("./CLIPcheckpoints/fine_tuned_clip_model")
#     processor = CLIPProcessor.from_pretrained("./CLIPcheckpoints/fine_tuned_clip_processor")

# # Example usage
# if __name__ == "__main__":
#     # Replace with your actual image paths and corresponding text descriptions
#     prefix = "../../datasets/fashion_dataset/rcnn_cropped_images/"
#     images = set([file for file in os.listdir(prefix) if "crop" in file])
#     dataset = pd.read_csv("../../datasets/fashion_dataset/dataset.csv")
#     # image_paths = [ for i in range(len(dataset))]
#     # texts = [ for i in range(len(dataset))]
#     pairs = [(dataset.iloc[i]['query'],prefix + str(dataset.iloc[i]['id'])+"_crop_1.jpg") 
#              for i in range(len(dataset)) if str(dataset.iloc[i]['id'])+"_crop_1.jpg" in images]
#     print(len(pairs))
#     print(pairs)
#     image_paths = [i[1] for i in pairs]
#     texts = [i[0] for i in pairs]
    
#     fine_tuned_model, fine_tuned_processor = fine_tune_clip(image_paths, texts)

#     # Test if you are able to load the checkpoints
#     load_checkpoints()
#     print("Checkpoints loaded!")
