import torch
from torchvision import transforms
from torch.utils.data import Dataset
import os

from src.utils import numpy_utils


class RetrievedDataset(Dataset):
    def __init__(self, files, embeddings_dir_path, mode = 'train', K = 10, precision_k = 3, transform = transforms.Compose([transforms.ToTensor()]) , target_transform = None):
        self.files = files
        self.embeddings_dir_path = embeddings_dir_path
        self.mode = mode

        # Number of items to be considered for training
        # Only used if mode is 'train'
        self.K = K
        
        # Precision for validation dataset
        # Only used if mode is 'val'
        self.precision = 0
        self.precision_k = precision_k

        self.transform = transform
        self.target_transform = target_transform

        # Read embeddings
        self.nd = numpy_utils.NumpyDecoder()

        # Creating list of (embedding, label) pairs
        self.samples = []
        for file in files:

            # For every query you will have multiple embedding files (query_id, top-1 img id, .., top-k img id)
            # Now for each query we will create multiple samples (query_id, query_emb, top-1 emb, top-1 custom features embs)
            # In get_item we need to open the corresponding emb file and do required transformations and send them (based on the id)
            
            # Requirements:
            # - Query has a format like 1234.txt
            # - Shouldn't have more than K lines (e.g K = 10)
            # - Items in the text file should be listed predicted relevance order

            # Implementation details:
            # - Assigning binary scores (could try custom scores) - i.e GT item gets 1 and rest 0
            # print(os.path.basename(file))
            query_id = os.path.basename(file).split('.')[0]
            query_samples = []
            recommendations = []
            with open(file, 'r') as f:
                for img_id in f:
                    img_id = img_id.strip()
                    recommendations.append(img_id)
                    if img_id == query_id:
                        query_samples.append([query_id, img_id.split('.')[0], 1])
                    else:
                        query_samples.append([query_id, img_id.split('.')[0], 0])

            if self.mode == 'train':
                query_samples = sorted(query_samples, key = lambda x : x[2], reverse = True) # Sort based on the relevance score
                
                # Consider only K samples for training
                query_samples = query_samples[:K]
                self.samples.extend(query_samples)
                continue
            
            elif self.mode == 'test' or self.mode == 'val':
                items = []
                relevance_scores = []
                for sample in query_samples:
                    items.append(sample[1])
                    relevance_scores.append(sample[2])

                self.samples.append({'query_id': query_id, 'item_ids': items, 'relevance_scores': relevance_scores})

                if query_id in recommendations[:self.precision_k]:
                    self.precision += 1

                continue
        
        if self.mode == 'val':
            print('Precision@' + str(self.precision_k) + ' for validation dataset before re-ranking:', self.precision/len(files))
        
        return

    def __len__(self):
        return len(self.samples)
    
    # Get embedding for item
    def get_item_emb(self, item_id):
        item_embedding_path = os.path.join(self.embeddings_dir_path, 'items', item_id + '.emb')
        item_emb = torch.tensor(self.nd.get_embeddings(item_embedding_path)[0], dtype = torch.float32)

        if self.mode == 'train':
            # Add custom feature embeddings for the item
            custom_feat_emb = torch.zeros_like(item_emb)
            custom_feat_dir_path = os.path.join(self.embeddings_dir_path, 'custom_features', item_id)
            for custom_feat_name in os.listdir(custom_feat_dir_path):
                custom_feat_path = os.path.join(custom_feat_dir_path, custom_feat_name)
                custom_feat_emb += torch.tensor(self.nd.get_embeddings(custom_feat_path)[0], dtype = torch.float32)

            # Adding item and custom features embeddings
            item_emb += custom_feat_emb
        
        item_emb = item_emb.flatten()
        return item_emb
    
    # Get embedding for query
    def get_query_emb(self, query_id):
        # Query embedding
        query_embedding_path = os.path.join(self.embeddings_dir_path, 'queries', query_id + '.emb')
        query_emb = torch.tensor(self.nd.get_embeddings(query_embedding_path)[0], dtype = torch.float32)
        query_emb = query_emb.flatten()
        return query_emb

    def __getitem__(self, index):
        sample = self.samples[index]

        if self.mode == 'train':
            query_id, item_id, relevance_score = sample
            
            query_emb = self.get_query_emb(query_id)
            item_emb = self.get_item_emb(item_id)
            emb = torch.cat((query_emb, item_emb), dim = 0)
            
            return {
                'query_ids': query_id,
                'embs': emb,
                'relevance_scores': torch.tensor(relevance_score, dtype = torch.int64)
            }

        elif self.mode == 'test' or self.mode == 'val':
            query_id = sample['query_id']
            item_ids = sample['item_ids']
            relevance_scores = sample['relevance_scores']

            query_emb = self.get_query_emb(query_id)

            # Item embeddings
            embs = []
            for i in range(0, len(item_ids)):
                item_id = item_ids[i]
                item_emb = self.get_item_emb(item_id)
                
                # TODO: Check dimensions
                emb = torch.cat((query_emb, item_emb), dim = 0)
                embs.append(emb)

            numerical_item_list = [int(item) for item in item_ids]
            return {
                # TODO: Make this query_id and item_id int/float instead of string
                'query_ids': query_id, 
                'item_ids': torch.tensor(numerical_item_list, dtype = torch.int64),
                'embs': torch.stack(embs, dim = 0),
                'relevance_scores': torch.tensor(relevance_scores, dtype = torch.int64)
            }
    
# if __name__ == 'main':
# retrieved_dataset = RetrievedDataset('../retrieved_items', 'embeddings/test_dataset')