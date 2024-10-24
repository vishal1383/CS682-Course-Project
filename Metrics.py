import NumpyUtils
import numpy as np
import os
from GenerateEmbeddings import ROOT_EMBEDDINGS_FOLDER

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
        
        self.metrics_path = os.path.join(ROOT_METRICS_FOLDER, self.dataset_paths[dataset_type])
        os.makedirs(self.metrics_path, exist_ok = True)
        
        self.embeddings_path = os.path.join(ROOT_EMBEDDINGS_FOLDER, self.dataset_paths[dataset_type])
        self.similarity_matrix = np.zeros((num_examples, top_k))
        if compute_sim:
            self.compute_similarity()
        else:
             self.similarity_matrix = np.load(os.path.join(self.metrics_path, 'similarity-matrix.npy'))
        return

    def cosine_sim(self, a: list, b: list):
        cos_sim = np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))
        return cos_sim
    
    def compute_similarity(self):
        nd = NumpyUtils.NumpyDecoder()
        texts, images = [],[]
        for i in range(self.num_examples):
            text, image = nd.get_embeddings(os.path.join(self.embeddings_path, 'texts' + str(i)))[0][0], nd.get_embeddings(os.path.join(self.embeddings_path, 'images' + str(i)))[0][0]
            texts.append(text)
            images.append(image)
        
        for i in range(len(texts)):
            sims = [self.cosine_sim(texts[i], images[j]) for j in range(len(images))]
            sims = np.argsort(sims)[::-1][:self.top_k]
            self.similarity_matrix[i] = np.asarray(sims)
            print(f"Done {i}")
        
        np.save(os.path.join(self.metrics_path, 'similarity-matrix.npy'),  self.similarity_matrix)

    def compute_precision(self):
        pass

    def compute_recall(self):
        # for every text query
        sim = np.load(os.path.join(self.metrics_path, 'similarity-matrix.npy'))
        cnt = sum([(i in sim[i]) for i in range(sim.shape[0])])/sim.shape[0]
        return cnt
    
    def get_misclassifications(self):
        indices = np.arange(0, self.similarity_matrix.shape[0])
        results = np.any(self.similarity_matrix == indices[:, None], axis = 1)
        misclassified_examples = np.where(results == False)[0]

        for i in range(0, misclassified_examples.shape[0]):
            print('Recommended examples for query ' + str( misclassified_examples[i]) + ' : ')
            for recommended_index in self.similarity_matrix[i]:
                 print(recommended_index)

# if __name__ == '__main__':
#     metrics = Metrics('fashion_dataset',  num_examples = 100, compute_sim = True)
#     print(metrics.compute_recall())
#     print(metrics.get_misclassifications())