import NumpyUtils
import numpy as np
import os
from GenerateEmbeddings import ROOT_EMBEDDINGS_FOLDER
from Datasets import *
from PlotUtils import PlotUtils

ROOT_METRICS_FOLDER = './metrics'

class Metrics():
    def __init__(self, dataset_type, num_examples = 10, top_k = 10, compute_sim = True): # Defining it in the outset is the easiest way to do that
        self.bs = 1
        self.top_k = top_k
        self.num_examples = num_examples

        self.type = dataset_type
        self.dataset_paths = {
            'deep_fashion': 'fashion_dataset',
            'test_data': 'dataset'
        }
        
        self.dataset_prefix = os.path.join(ROOT_DATASET_FOLDER, self.dataset_paths[dataset_type])
        self.data_df = pd.read_csv(os.path.join(self.dataset_prefix, 'dataset.csv'))
        ids = self.data_df['id'].tolist()[:self.num_examples]
        # Map indices to actual ids of the samples
        self.indices_to_ids = {index: value for index, value in enumerate(ids)}
        
        self.metrics_path = os.path.join(ROOT_METRICS_FOLDER, self.dataset_paths[dataset_type])
        os.makedirs(self.metrics_path, exist_ok = True)
        
        self.embeddings_path = os.path.join(ROOT_EMBEDDINGS_FOLDER, self.dataset_paths[dataset_type])
        self.similarity_matrix = np.zeros((num_examples, top_k))

        self.similarity_matrix_path = os.path.join(self.metrics_path, 'similarity-matrix.npy')
        if compute_sim:
            self.compute_similarity()
        else:
             self.similarity_matrix = np.load(self.similarity_matrix_path)
        
        self.plotUtils = PlotUtils()
        return

    def cosine_sim(self, a: list, b: list):
        cos_sim = np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))
        return cos_sim
    
    def compute_similarity(self):
        nd = NumpyUtils.NumpyDecoder()
        texts, images = [],[]
        for i in range(self.num_examples):
            id = self.indices_to_ids[i]
            text_embedding_path = os.path.join(self.embeddings_path, 'texts_' + str(id))
            image_embedding_path = os.path.join(self.embeddings_path, 'images_' + str(id))

            text, image = nd.get_embeddings(text_embedding_path)[0], nd.get_embeddings(image_embedding_path)[0]
            texts.append(text)
            images.append(image)
        
        for i in range(len(texts)):
            id = self.indices_to_ids[i]
            sims = [self.cosine_sim(texts[i], images[j]) for j in range(len(images))]
            self.similarity_matrix[i] = np.argsort(sims)[::-1][:self.top_k]
            print(f"Done {id}")

        np.save(self.similarity_matrix_path,  self.similarity_matrix)

    def compute_precision(self):
        pass

    def compute_recall(self):
        # for every text query
        sim = np.load(self.similarity_matrix_path).astype(int)
        cnt = sum([(i in sim[i]) for i in range(sim.shape[0])])/sim.shape[0]
        return cnt
    
    # Generates 'num_examples' recommendations for a give text query for different scenarios:
    # 1. Incorrect recommendations
    # 2. Correct recommendations
    # 3. Recommendations with recall 1
    def get_recommendations(self, num_examples = 2):
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

        print('Getting the incorrect recommendations...')
        for i in incorrect_examples[:num_examples]:
            text = texts[i]
            
            recommended_images = []
            for index in self.similarity_matrix[i]:
                 recommended_images.append(Image.open(self.image_paths[int(index)]))

            ground_truth_image = Image.open(self.image_paths[i])
            self.plotUtils.plot_recommendations(text, recommended_images, [False] * len(recommended_images) + [True], ground_truth_image)
        
        print('Getting the correct recommendations..')
        for i in correct_examples[:num_examples]:
            text = texts[i]
            
            recommended_images = []
            labels = np.where(self.similarity_matrix[i] != i, False, True)
            for index in self.similarity_matrix[i]:
                 recommended_images.append(Image.open(self.image_paths[int(index)]))

            self.plotUtils.plot_recommendations(text, recommended_images, labels)

        print('Getting the recommendations with recall 1..')
        for i in examples_with_recall_1[:num_examples]:
            text = texts[i]
            
            recommended_images = []
            labels = np.where(self.similarity_matrix[i] != i, False, True)
            for index in self.similarity_matrix[i]:
                 recommended_images.append(Image.open(self.image_paths[int(index)]))

            self.plotUtils.plot_recommendations(text, recommended_images, labels)

# if __name__ == '__main__':
#     metrics = Metrics('fashion_dataset',  num_examples = 100, compute_sim = True)
#     print(metrics.compute_recall())
#     print(metrics.get_misclassifications())

'''
    TODOs:
    1. Similarity matrix doesn't contain sim values - just has the indices -> could be changed
'''