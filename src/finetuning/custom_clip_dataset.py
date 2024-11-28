from PIL import Image
from torch.utils.data import Dataset
import os
os.environ['CUDA_HOME'] = os.getenv('CONDA_PREFIX')


dataset_paths = {
    'deep_fashion': 'fashion_dataset',
    'test_data': 'dataset'
}

class CustomCLIPDataset(Dataset):
    def __init__(self, images, texts, processor, max_length = 77):
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
    
