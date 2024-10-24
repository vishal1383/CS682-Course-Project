import os
import pandas as pd
import GenerateEmbeddings
from GenerateEmbeddings import *
from NumpyUtils import *

class Datasets:
    def __init__(self, g: GenerateEmbeddings, n_examples, dataset_type):
        self.images = [] # Expected to be a Image.open file
        self.texts = []  # expected to be a file

        self.image_tensors = []
        self.text_tensors = []
        self.type = dataset_type
        self.dataset_paths = {
            'deep-fashion': './fashion-dataset/',
            'test-deep-fashion': 'dataset'
        }
        self.prefix = self.dataset_paths[dataset_type]
        self.embbeding_util = g
        self.n_examples = n_examples
        self.embedding_prefix = os.path.join(ROOT_EMBEDDINGS_FOLDER, self.prefix)
        os.makedirs(self.embedding_prefix, exist_ok = True)
    
    def load_dataset(self,):
        if self.type == 'deep-fashion' or self.type == 'test-deep-fashion':
            image_data = pd.read_csv(os.path.join(self.prefix, 'images.csv'))
            image_filenames = [image_data.iloc[i]['filename'] for i in range(len(image_data))][:self.n_examples]

            text_data_df = pd.read_csv(os.path.join(self.prefix, 'styles.csv'))
            self.images = [os.path.join(self.prefix, 'images', image_name) for image_name in image_filenames]

            text_data_df['text_data'] = text_data_df.iloc[:, 1:].apply(lambda row: ' '.join(row.astype(str)), axis = 1)
            self.texts = text_data_df['text_data'].to_list()

            # Generate embeddings for the tensors
            for i in range(len(self.images)):
                embs = self.embbeding_util.generate_embeddings([self.texts[i]], [Image.open(self.images[i])])
                generateEmbeddings.save_embeddings([embs[1]], os.path.join(self.embedding_prefix, 'images' + str(i)))
                generateEmbeddings.save_embeddings([embs[0]], os.path.join(self.embedding_prefix, 'texts' + str(i)))
                print(f"Done {i}")
        else:
            raise ValueError(f"Dataset {self.type} not supported")

if __name__ == '__main__':
    generateEmbeddings = GenerateEmbeddings(data_path ='./destination')
    # d = Datasets(generateEmbeddings, 10000,'deep-fashion')
    d = Datasets(generateEmbeddings, 10000,'test-deep-fashion')
    d.load_dataset()