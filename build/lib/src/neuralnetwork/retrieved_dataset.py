import torch
from torchvision import transforms
from torch.utils.data import Dataset
import os
import warnings
from utils import numpy_utils

class RetrievedDataset(Dataset):
    def __init__(self, path, embeddings_dir_path, transform = transforms.Compose([transforms.ToTensor()]) , target_transform = None):
        self.path = path
        self.embeddings_dir_path = embeddings_dir_path

        self.transform = transform
        self.target_transform = target_transform

        # read embeddings
        self.nd = numpy_utils.NumpyDecoder()

        # creating list of (embedding, label) pairs
        self.samples = []
        for query in os.listdir(path):
             # Ignore hidden files/folders
            if query.startswith('.'):
                warning = "Warning: Ignoring hidden files/folders - " + query
                warnings.warn(warning)
                continue

            # For every query you will have multiple embedding files (query_id, top-1 img id, .., top-k img id)
            # Now for each query we will create multiple samples (query_id, query_emb, top-1 emb, top-1 custom features embs)
            # In get_item we need to open the corresponding emb file and do required transformations and send them (based on the id)
            
            # Requirements:
            # - Query has a format like 1234.txt
            # - Shouldn't have more than K lines (e.g K = 10)
            # - Items in the text file should be listed predicted relevance order

            # Implementation details:
            # - Assigning binary scores (could try custom scores) - i.e GT item gets 1 and rest 0
            query_id = query.split('.')[0]
            with open(os.path.join(path, query), 'r') as f:
                for img_id in f:
                    if img_id == query_id:
                        self.samples.append([query_id, img_id.strip().split('.')[0], 1.0])
                    else:
                        self.samples.append([query_id, img_id.strip().split('.')[0], 0.0])

        print('self.samples', self.samples)
        print('self.samples', self.samples)


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        query_id, item_id, score = self.samples[index]
        
        # Query embedding
        query_embedding_path = os.path.join(self.embeddings_dir_path, 'queries', query_id)
        t = self.nd.get_embeddings(query_embedding_path)[0]
        print(t, type(t))
        query_emb = torch.tensor(self.nd.get_embeddings(query_embedding_path)[0])

        # Item embedding
        item_embedding_path = os.path.join(self.embeddings_dir_path, 'items', item_id)
        item_emb = self.nd.get_embeddings(item_embedding_path)[0]
        print(type(item_emb), item_emb.shape)
        item_emb = torch.tensor(self.nd.get_embeddings(item_embedding_path)[0])
        print(type(item_emb), item_emb.shape)
        
        # Add custom feature embeddings for the item
        custom_feat_emb = torch.empty_like(item_emb)
        custom_feat_dir_path = os.path.join(self.embeddings_dir_path, 'custom_features', item_id)
        for custom_feat_path in os.listdir(custom_feat_dir_path):
            custom_feat_emb += torch.tensor(self.nd.get_embeddings(custom_feat_path)[0])

        # Adding item and custom features embeddings
        item_emb += custom_feat_emb

        query_emb = query_emb.flatten()
        item_emb = item_emb.flatten()
        # Check dimensions
        emb = torch.cat((query_emb, item_emb), dim = 0)

        return {
            'query_id': query_id,
            'emb': emb,
            'score': score
        }
    
# if __name__ == 'main':
# retrieved_dataset = RetrievedDataset('../retrieved_items', 'embeddings/test_dataset')