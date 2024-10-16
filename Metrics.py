# import Datasets
import NumpyUtils
import numpy as np
import os
# from Datasets import *

class Metrics():
    def __init__(self, metrics_path: str, num_examples=10, batch_size=1, top_k=10, load_sim=True): # Defining it in the outset is the easiest way to do that
        self.bs = 1
        self.top_k = top_k
        self.similarity_matrix = np.zeros((num_examples, top_k))
        self.metrics_path = metrics_path
        if load_sim:
            self.compute_similarity()
    def cosine_sim(self, a: list, b: list):
        cos_sim = np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))
        return cos_sim
    def compute_similarity(self, ):
        os.makedirs('./Metrics/', exist_ok=True) 
        prefix =  self.metrics_path
        nd = NumpyUtils.NumpyDecoder()
        texts, images = nd.get_embeddings(prefix + 'texts'), nd.get_embeddings(prefix + 'images')
        for i in range(len(texts)):
            sims = [self.cosine_sim(texts[i][0], images[j][0]) for j in range(len(images))]
            sims = np.argsort(sims)[::-1][:self.top_k]
            self.similarity_matrix[i] = np.asarray(sims)
        np.save(self.metrics_path + 'similarity-matrix.npy',  self.similarity_matrix)

    def compute_precision(self,):
        pass

    def compute_recall(self,):
        sim = np.load(self.metrics_path + 'similarity-matrix.npy')
        cnt=sum([(i+1 in sim[i]) for i in range(sim.shape[0])])/sim.shape[0]
        return cnt
        
        
        

if __name__ == '__main__':
    metrics = Metrics('./Embeddings/fashion-dataset/', load_sim=False)
    print(metrics.compute_recall())