from Datasets import Datasets
from Metrics import Metrics

if __name__ == "__main__":
    # Clean & load the dataset (embeddings)
    dataset = Datasets('deep_fashion', num_examples = 5000)
    dataset.clean_dataset()
    dataset.load_dataset()

    # Compute metrics
    metrics_10 = Metrics('deep_fashion',  num_examples = 100, top_k = 10, compute_sim = True)
    metrics_30 = Metrics('deep_fashion',  num_examples = 100, top_k = 30, compute_sim = True)
    metrics_50 = Metrics('deep_fashion',  num_examples = 100, top_k = 50, compute_sim = True)
    
    print(metrics_10.get_recommendations(num_examples = 2))
    print(metrics_30.get_recommendations(num_examples = 2))
    print(metrics_50.get_recommendations(num_examples = 2))
'''
    TODOs:
    1. Pass args from CLI to run required modules (not all)
'''