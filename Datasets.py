import os
import pandas as pd
import GenerateEmbeddings
from GenerateEmbeddings import *
from NumpyUtils import *

ROOT_DATASET_FOLDER = './datasets/'

class Datasets:
    def __init__(self, dataset_type, num_examples):
        self.n_examples = num_examples
        
        self.images = [] # Expected to be a Image.open file
        self.texts = []  # expected to be a file
        
        self.type = dataset_type
        self.dataset_paths = {
            'deep_fashion': 'fashion_dataset',
            'test_data': 'dataset'
        }
        
        self.embedding_prefix = os.path.join(ROOT_EMBEDDINGS_FOLDER, self.dataset_paths[dataset_type])
        os.makedirs(self.embedding_prefix, exist_ok = True)
        self.embbeding_util = GenerateEmbeddings(self.embedding_prefix)
        
        self.dataset_prefix = os.path.join(ROOT_DATASET_FOLDER, self.dataset_paths[dataset_type])
        os.makedirs(self.dataset_prefix, exist_ok = True)

    # Merges images and text csv to a single file
    # Creates a single csv named dataset.csv
    def clean_dataset(self):
        if self.type == 'deep_fashion' or self.type == 'test_data':
            images_df = pd.read_csv(os.path.join(self.dataset_prefix, 'images.csv'))
            images_df['id'] = images_df['filename'].apply(lambda x: x.replace('.jpg', ' ')).astype(int)
            
            text_df = pd.read_csv(os.path.join(self.dataset_prefix, 'styles.csv'), on_bad_lines = 'skip')

            data_df = text_df.merge(images_df, on = 'id', how = 'left').reset_index(drop = True)
            data_df = data_df[:self.n_examples]
            data_df.to_csv(os.path.join(self.dataset_prefix, 'dataset.csv'))
        else:
            raise ValueError(f"Dataset {self.type} not supported")
        
        return
    
    def load_dataset(self):
        if self.type == 'deep_fashion' or self.type == 'test_data':
            data_df = pd.read_csv(os.path.join(self.dataset_prefix, 'dataset.csv'))
            
            image_filenames = [data_df.iloc[i]['filename'] for i in range(len(data_df))]
            self.images = [os.path.join(self.dataset_prefix, 'images', image_name) for image_name in image_filenames]

            data_df['text_data'] = data_df.iloc[:, 1:].apply(lambda row: ' '.join(row.astype(str)), axis = 1)
            self.texts = data_df['text_data'].to_list()

            # Generate embeddings for the tensors
            for i in range(len(self.images)):
                embs = self.embbeding_util.generate_embeddings([self.texts[i]], [Image.open(self.images[i])])
                self.embbeding_util.save_embeddings([embs[1]], os.path.join('images' + str(i)))
                self.embbeding_util.save_embeddings([embs[0]], os.path.join('texts' + str(i)))
                print(f"Done {i}")
        else:
            raise ValueError(f"Dataset {self.type} not supported")

# if __name__ == '__main__':
#     d = Datasets('deep_fashion', 100)
#     d.clean_dataset()
#     d.load_dataset()