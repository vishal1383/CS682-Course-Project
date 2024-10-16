import json
import numpy as np

class NumpyDecoder(json.JSONDecoder):
    def __init__(self, *args, **kwargs):
        super().__init__(object_hook=self.object_hook, *args, **kwargs)

    def object_hook(self, obj):
        if '__ndarray__' in obj:
            return np.array(obj['__ndarray__'])
        return obj

def load_embeddings(embeddings_path):
    with open(embeddings_path, 'r') as file:
        json_data = file.read()
    embeddings = json.loads(json_data, cls=NumpyDecoder)
    return embeddings

print(load_embeddings('texts')[1][0])