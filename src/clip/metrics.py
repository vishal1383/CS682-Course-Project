import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
import os
import random

from sklearn.model_selection import train_test_split

from src.clip.generate_embeddings import ROOT_EMBEDDINGS_FOLDER
from src.clip.raw_dataset import *
from src.utils.plot_utils import PlotUtils
from src.utils import numpy_utils

ROOT_METRICS_FOLDER = '../metrics'

class Metrics():
    def __init__(self, dataset_type, num_examples = 10, top_k = 10, compute_sim = True): # Defining it in the outset is the easiest way to do that
        self.bs = 1
        self.top_k = top_k
        self.num_examples = num_examples
        self.model_type = 'clip'
        self.dataset_paths = {
            'deep_fashion': 'fashion_dataset',
            'test_data': 'dataset'
        }
        
        self.dataset_prefix = os.path.join(ROOT_DATASET_FOLDER, self.dataset_paths[dataset_type])
        self.data_df = pd.read_csv(os.path.join(self.dataset_prefix, 'dataset.csv'))
        ids = self.data_df['id'].tolist()[:self.num_examples]

        seed = 42
        random.seed(seed)
        random.shuffle(ids)
        self.train_ids, temp_ids = train_test_split(ids, train_size = 0.7, random_state = seed)
        self.val_ids, self.test_ids = train_test_split(temp_ids, test_size = 0.15 / (0.15 + 0.15), random_state = seed)
        # print('train val test ids', self.train_ids, self.val_ids, self.test_ids)
        print('train val test ids', self.val_ids)

        # Map indices to actual ids of the samples
        self.indices_to_ids = {index: value for index, value in enumerate(ids)}
        self.ids_to_indices = {value: index for index, value in enumerate(ids)}

        self.retrieved_results_path = os.path.join('../retrieved_items', self.dataset_paths[dataset_type], str(top_k))
        os.makedirs(self.retrieved_results_path, exist_ok = True)
        
        self.embeddings_path = os.path.join(ROOT_EMBEDDINGS_FOLDER, self.dataset_paths[dataset_type], self.model_type)
        self.embedding_query_prefix = os.path.join(self.embeddings_path, 'queries')
        self.embedding_item_prefix = os.path.join(self.embeddings_path, 'items')

        self.metrics_path = os.path.join(ROOT_METRICS_FOLDER, self.dataset_paths[dataset_type], self.model_type)
        os.makedirs(self.metrics_path, exist_ok = True)
        
        self.similarity_matrix = np.zeros((num_examples, top_k))
        self.similarity_matrix_path = os.path.join(self.metrics_path, 'similarity-matrix_' + str(self.top_k) + '.npy')
        if compute_sim:
            self.compute_similarity()
        else:
             self.similarity_matrix = np.load(self.similarity_matrix_path)
        
        self.plotUtils = PlotUtils()
        return
    
    def compute_similarity(self):
        print('\nComputing similarity matrix for all image-text pairs')
        nd = numpy_utils.NumpyDecoder()
        
        texts = np.zeros((self.num_examples, EMBEDDING_DIM))
        images = np.zeros((self.num_examples, EMBEDDING_DIM))
        texts, images = [],[]
        for i in range(self.num_examples):
            id = self.indices_to_ids[i]
            text_embedding_path = os.path.join(self.embedding_query_prefix, str(id) + '.emb')
            image_embedding_path = os.path.join(self.embedding_item_prefix, str(id) + '.emb')

            text, image = nd.get_embeddings(text_embedding_path)[0], nd.get_embeddings(image_embedding_path)[0]
            texts.append(text)
            images.append(image)

        self.similarity_matrix_orig = cosine_similarity(texts, images)
        self.similarity_matrix = np.argsort(-self.similarity_matrix_orig, axis = 1)[:, :self.top_k]

        np.save(self.similarity_matrix_path,  self.similarity_matrix)
        print('Saved similarity matrix at ' + self.similarity_matrix_path)
        print('\n' + '-' * 50)
        return

    def compute_recall(self):
        # TODO: Might have to write to a file
        # for every text query - Entire dataset
        sim = np.load(self.similarity_matrix_path).astype(int)
        recall = sum([(i in sim[i]) for i in range(sim.shape[0])])/sim.shape[0]
        print(f'\nRecall@{self.top_k} for Entire dataset with {self.num_examples} examples is: ', recall)
        print('\n' + '-' * 50)

        # Train data
        train_indices = [self.ids_to_indices[id] for id in self.train_ids]
        similarity_matrix_train = np.argsort(-self.similarity_matrix_orig[train_indices, :][:, train_indices], axis = 1)[:, :self.top_k]
        recall_train = sum([(i in similarity_matrix_train[i]) for i in range(similarity_matrix_train.shape[0])])/similarity_matrix_train.shape[0]
        print(f'\nRecall@{self.top_k} for Train dataset with {len(train_indices)} examples is: ', recall_train)
        print('\n' + '-' * 50)

        # Val data
        val_indices = [self.ids_to_indices[id] for id in self.val_ids]
        similarity_matrix_val = np.argsort(-self.similarity_matrix_orig[val_indices, :][:, val_indices], axis = 1)[:, :self.top_k]
        recall_val = sum([(i in similarity_matrix_val[i]) for i in range(similarity_matrix_val.shape[0])])/similarity_matrix_val.shape[0]
        print(f'\nRecall@{self.top_k} for Val dataset with {len(val_indices)} examples is: ', recall_val)
        print('\n' + '-' * 50)

        # Test data
        test_indices = [self.ids_to_indices[id] for id in self.test_ids]
        similarity_matrix_test = np.argsort(-self.similarity_matrix_orig[test_indices, :][:, test_indices], axis = 1)[:, :self.top_k]
        recall_test = sum([(i in similarity_matrix_test[i]) for i in range(similarity_matrix_test.shape[0])])/similarity_matrix_test.shape[0]
        print(f'\nRecall@{self.top_k} for Test dataset with {len(test_indices)} examples is: ', recall_test)
        print('\n' + '-' * 50)

        return recall
    
    # Save for all examples (if GT is present in top_k or not)
    def save_recommendations(self):
        indices = np.arange(0, self.similarity_matrix.shape[0])
        results = np.any(self.similarity_matrix == indices[:, None], axis = 1)
        correct_examples = np.where(results == True)[0]
        incorrect_examples = np.where(results == False)[0]

        recommendations = np.vectorize(self.indices_to_ids.get)(self.similarity_matrix)

        # TODO: they wil retrieved in ascending order
        # query_ids = np.array((self.indices_to_ids.values())[correct_examples]
        
        for i, row in enumerate(recommendations):
            type = None
            if i in correct_examples:
                type = '_c'
            elif i in incorrect_examples:
                type = '_ic'

            filename = os.path.join(self.retrieved_results_path, str(self.indices_to_ids[i]) + type + ".txt")
            with open(filename, 'w') as f:
                f.write("\n".join(map(str, row)))
        
        return
    
    # Generates 'num_examples' recommendations for a give text query for different scenarios:
    # 1. Incorrect recommendations
    # 2. Correct recommendations
    # 3. Recommendations with recall 1
    def get_recommendations(self, num_examples = 2):
        print('\nGeting the recommendations from the model:')
        indices = np.arange(0, self.similarity_matrix.shape[0])
        
        results = np.any(self.similarity_matrix == indices[:, None], axis = 1)        
        incorrect_examples = np.where(results == False)[0]
        correct_examples = np.where(results == True)[0]
        
        first_recommendations = self.similarity_matrix[:, 0].reshape(-1, 1)
        results = np.any(first_recommendations == indices[:, None], axis = 1)
        examples_with_recall_1 = np.where(results == True)[0]
        
        texts = self.data_df['query'].to_list()[:self.num_examples]

        image_filenames = [self.data_df.iloc[i]['filename'] for i in range(self.num_examples)]
        self.image_paths = [os.path.join(self.dataset_prefix, 'images', image_name) for image_name in image_filenames]
        
        if len(incorrect_examples) == 0:
            print('|', '-' * 10, 'No incorrect recommendations found..!')
        else:
            print('|', '-' * 10, 'Getting the incorrect recommendations...')
            # If we don't have num_examples incorrect examples, return whatever we have
            examples = np.random.choice(incorrect_examples, size = min(len(incorrect_examples), num_examples), replace = False)
            # examples = incorrect_examples[:num_examples]
            for i in examples:
                text = texts[i] + f'({self.indices_to_ids[i]})'
                
                recommended_images = {'ids': [], 'imgs': []}
                for index in self.similarity_matrix[i]:
                    recommended_images['ids'].append(self.indices_to_ids[int(index)])
                    recommended_images['imgs'].append(Image.open(self.image_paths[int(index)]))

                ground_truth_images = {'ids': [self.indices_to_ids[i]], 'imgs': [Image.open(self.image_paths[i])]}
                self.plotUtils.plot_recommendations(text, recommended_images, [False] * self.top_k, ground_truth_images)
        
        if len(correct_examples) == 0:
            print('|', '-' * 10, 'No correct recommendations found..!')
        else:
            print('|', '-' * 10, 'Getting the correct recommendations..')
            examples = np.random.choice(correct_examples, size = min(len(correct_examples), num_examples), replace = False)
            # examples = correct_examples[:num_examples]
            for i in examples:
                text = texts[i] + f'({self.indices_to_ids[i]})'
                
                recommended_images = {'ids': [], 'imgs': []}
                labels = np.where(self.similarity_matrix[i] != i, False, True)
                for index in self.similarity_matrix[i]:
                    recommended_images['ids'].append(self.indices_to_ids[int(index)])
                    recommended_images['imgs'].append(Image.open(self.image_paths[int(index)]))

                self.plotUtils.plot_recommendations(text, recommended_images, labels)

        if len(examples_with_recall_1) == 0:
            print('|', '-' * 10, 'No recommendations with recall 1 found..!')
        else:
            print('|', '-' * 10, 'Getting the recommendations with recall 1..')
            examples = np.random.choice(examples_with_recall_1, size = min(len(examples_with_recall_1), num_examples), replace = False)
            # examples = examples_with_recall_1[:num_examples]
            for i in examples:
                text = texts[i] + f'({self.indices_to_ids[i]})'
                
                recommended_images = {'ids': [], 'imgs': []}
                labels = np.where(self.similarity_matrix[i] != i, False, True)
                for index in self.similarity_matrix[i]:
                    recommended_images['ids'].append(self.indices_to_ids[int(index)])
                    recommended_images['imgs'].append(Image.open(self.image_paths[int(index)]))

                self.plotUtils.plot_recommendations(text, recommended_images, labels)
        
        print('\n' + '-' * 50)
        return

# if __name__ == '__main__':
#     metrics = Metrics('fashion_dataset',  num_examples = 100, compute_sim = True)
#     print(metrics.compute_recall())
#     print(metrics.get_misclassifications())

'''
    TODOs:
    1. Similarity matrix doesn't contain sim values - just has the indices -> could be changed
    2. Correct examples plot can have images without any +ve pairs - happens because we are only showing top_k and if the ground_truth might not be in top_k
'''