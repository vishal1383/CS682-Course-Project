import json
import numpy as np
import torch
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.ndarray, torch.Tensor)):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
    
class NumpyDecoder(json.JSONDecoder):
    def __init__(self, *args, **kwargs):
        super().__init__(object_hook=self.object_hook, *args, **kwargs)

    def object_hook(self, obj):
        if '__ndarray__' in obj:
            return np.array(obj['__ndarray__'])
        return obj
    
    def get_embeddings(self, embeddings_path):
        with open(embeddings_path, 'r') as file:
            json_data = file.read()
        embeddings = json.loads(json_data, cls=NumpyDecoder)
        return embeddings