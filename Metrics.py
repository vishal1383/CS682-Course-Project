import NumpyUtils
import numpy as np
import os
from GenerateEmbeddings import ROOT_EMBEDDINGS_FOLDER

class Metrics():
    def __init__(self, metrics_path: str, num_examples = 10, batch_size = 1, top_k = 10, load_sim = True): # Defining it in the outset is the easiest way to do that
        self.bs = 1
        self.top_k = top_k
        self.num_examples = num_examples
        self.similarity_matrix = np.zeros((num_examples, top_k))
        self.metrics_folder = metrics_path
        self.metrics_path = os.path.join(ROOT_EMBEDDINGS_FOLDER, metrics_path)
        if load_sim:
            self.compute_similarity()
        else:
             self.similarity_matrix = np.load(os.path.join(self.metrics_path, 'similarity-matrix.npy'))
    
    def cosine_sim(self, a: list, b: list):
        cos_sim = np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))
        return cos_sim
    
    def compute_similarity(self, ):
        os.makedirs(os.path.join('Metrics', self.metrics_folder), exist_ok=True)
        prefix =  self.metrics_path
        nd = NumpyUtils.NumpyDecoder()
        texts, images = [],[]
        for i in range(self.num_examples):
            text, image = nd.get_embeddings(os.path.join(prefix, 'texts' + str(i)))[0][0], nd.get_embeddings(os.path.join(prefix, 'images' + str(i)))[0][0]
            texts.append(text)
            images.append(image)
        
        for i in range(len(texts)):
            sims = [self.cosine_sim(texts[i], images[j]) for j in range(len(images))]
            sims = np.argsort(sims)[::-1][:self.top_k]
            self.similarity_matrix[i] = np.asarray(sims)
            print(f"Done {i}")
        np.save(os.path.join(self.metrics_path, 'similarity-matrix.npy'),  self.similarity_matrix)

    def compute_precision(self,):
        pass

    def compute_recall(self,):
        sim = np.load(os.path.join(self.metrics_path, 'similarity-matrix.npy'))
        cnt=sum([(i+1 in sim[i]) for i in range(sim.shape[0])])/sim.shape[0]
        return cnt
    
    def get_misclassifications(self):
        indices = np.arange(0, self.similarity_matrix.shape[0])
        results = np.any(self.similarity_matrix == indices[:, None], axis = 1)
        misclassified_examples = np.where(results == False)[0]

        for i in range(0, misclassified_examples.shape[0]):
            print('Recommended examples for query ' + str( misclassified_examples[i]) + ' : ')
            for recommended_indices in self.similarity_matrix[i]:
                 print(recommended_indices)

if __name__ == '__main__':
    # metrics = Metrics('fashion-dataset',  num_examples=5000, load_sim=False)
    metrics = Metrics('dataset',  num_examples = 15, load_sim = True)
    # print(metrics.compute_recall())
    print(metrics.get_misclassifications())