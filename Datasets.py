import pandas as pd
import numpy as np
import GenerateEmbeddings
from GenerateEmbeddings import *

import json
import numpy as np
import NumpyUtils
from NumpyUtils import *



class Datasets:
    def __init__(self, g: GenerateEmbeddings, n_examples, dataset_type):
        self.images = [] # Expected to be a Image.open file
        self.texts = []  # expected to be a file

        self.image_tensors = []
        self.text_tensors = []
        self.type = dataset_type
        self.dataset_paths = {
            'deep-fashion': './fashion-dataset/'
        }
        self.prefix = self.dataset_paths[dataset_type]
        self.embbeding_util = g
        self.n_examples = n_examples    
        self.embedding_prefix = ROOT_EMBEDDINGS_FOLDER + self.prefix
    def load_dataset(self,):
        if self.type == 'deep-fashion':
            image_data = pd.read_csv(self.prefix + 'images.csv')
            image_filenames = [image_data.iloc[i]['filename'] for i in range(len(image_data))][:self.n_examples]

            #text_data = pd.read_csv(self.prefix +'styles.csv')
            
            self.images = [self.prefix + 'images/' + image_name for image_name in image_filenames]
            
            text_data = []
            with open(self.prefix + 'styles.csv') as F:
                cnt=0
                for line in F.readlines():
                    if cnt>=self.n_examples:
                        break
                    text_data.append(" ".join(line.split(',')[1:]))
                    cnt+=1
            self.texts = text_data

            # Save the resulting embeddings
            embeddings_dir = ROOT_EMBEDDINGS_FOLDER

            # Generate embeddings for the tensors
            for i in range(len(self.images)):
                embs = self.embbeding_util.generate_embeddings([self.texts[i]], [Image.open(self.images[i])])
                # self.image_tensors.append(embs[1])
                # self.text_tensors.append(embs[0])

                generateEmbeddings.save_embeddings([embs[1]], self.embedding_prefix + 'images'+str(i))
                generateEmbeddings.save_embeddings([embs[0]], self.embedding_prefix + 'texts'+str(i))
                print(f"Done {i}")
            # for i in range(len(self.image_tensors)):
        else:
            raise ValueError(f"Dataset {self.type} not supported")


    # @staticmethod
    # def get_embeddings_Store():
    #     return self.embedding_prefix
if __name__ == '__main__':
    generateEmbeddings = GenerateEmbeddings(data_path ='./destination')
    d = Datasets(generateEmbeddings, 10000,'deep-fashion')
    d.load_dataset()
