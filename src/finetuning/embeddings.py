import os
import pandas as pd
import src.clip.generate_embeddings as GenerateEmbeddings
from src.clip.generate_embeddings import *
from src.utils.numpy_utils import *
from PIL import Image

ROOT_DATASET_FOLDER = '../datasets/'

class Emebeddings:
    def __init__(self, dataset_type, num_examples):
        self.n_examples = num_examples
         
        self.texts = []  # List of text queries
        self.image_paths = [] # List of paths of the corresponding images
        
        self.type = dataset_type
        self.dataset_paths = {
            'deep_fashion': 'fashion_dataset',
            'test_data': 'dataset'
        }
        self.model_type = 'finetune'
        
        # Create folders for dataset
        # - embeddings/{dataset}
        self.dataset_prefix = os.path.join(ROOT_DATASET_FOLDER, self.dataset_paths[dataset_type])
        os.makedirs(self.dataset_prefix, exist_ok = True)

        # Create folders for embeddings
        # - embeddings/{dataset}
        self.embedding_prefix = os.path.join(ROOT_EMBEDDINGS_FOLDER, self.dataset_paths[dataset_type], self.model_type)
        os.makedirs(self.embedding_prefix, exist_ok = True)
        self.embbeding_util = GenerateEmbeddings()
        
        # - embeddings/{dataset}/items
        self.embedding_query_prefix = os.path.join(self.embedding_prefix, 'queries')
        os.makedirs(self.embedding_query_prefix, exist_ok = True)
        
        # - embeddings/{dataset}/queries
        self.embedding_item_prefix = os.path.join(self.embedding_prefix, 'items')
        os.makedirs(self.embedding_item_prefix, exist_ok = True)

        # - embeddings/{dataset}/custom_features
        # TODO - Remove this from here (Will be populated in metrics.py)
        self.embedding_custom_feat_prefix = os.path.join(self.embedding_prefix, 'custom_features')
        os.makedirs(self.embedding_custom_feat_prefix, exist_ok = True)

        return
    
    def load_dataset(self):
        print(f'\nLoading {self.type} dataset and saving the embeddings for the text-image pairs')
        if self.type == 'deep_fashion' or self.type == 'test_data':
            data_df = pd.read_csv(os.path.join(self.dataset_prefix, 'dataset.csv'))
            
            # Ids in the dataset corresponding to image-text pair
            ids = data_df['id'].tolist()
            
            image_filenames = [data_df.iloc[i]['filename'] for i in range(len(data_df))]
            self.image_paths = [os.path.join(self.dataset_prefix, 'images', image_name) for image_name in image_filenames]

            self.texts = data_df['query'].to_list()

            # Generate and save embeddings for text-image pairs
            for i in range(len(self.texts)):
                embs = self.embbeding_util.generate_embeddings([self.texts[i]], [Image.open(self.image_paths[i])])
                self.embbeding_util.save_embeddings(embs['text_embs'], os.path.join(self.embedding_query_prefix, str(ids[i]) + '.emb'))
                self.embbeding_util.save_embeddings(embs['image_embeds'], os.path.join(self.embedding_item_prefix, str(ids[i]) + '.emb'))
                
                # TODO: If image is not present in custom_feats
                os.makedirs(os.path.join(self.embedding_custom_feat_prefix, str(ids[i])), exist_ok = True)
                if os.path.exists(self.custom_features_paths[i]):
                    custom_feat_embs = self.embbeding_util.generate_image_embedding([Image.open(self.custom_features_paths[i])])
                    self.embbeding_util.save_embeddings(custom_feat_embs, os.path.join(self.embedding_custom_feat_prefix, str(ids[i]), str(ids[i]) + '.emb'))

                if i == len(self.texts) - 1 or (i >= 100 and i % 100 == 0):
                    print('|', '-' * 10, 'Done processing ' + str(i + 1) + ' text-image pairs')
        else:
            raise ValueError(f"Dataset {self.type} not supported")
        
        print('\n' + '-' * 50)
        return

# if __name__ == '__main__':
#     d = Datasets('deep_fashion', 100)
#     d.clean_dataset()
#     d.load_dataset()