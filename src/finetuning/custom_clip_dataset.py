from PIL import Image
from torch.utils.data import Dataset
import os
os.environ['CUDA_HOME'] = os.getenv('CONDA_PREFIX')

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
    
