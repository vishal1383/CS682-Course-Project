from Datasets import Datasets
from Metrics import Metrics

if __name__ == "__main__":
    # Clean & load the dataset (embeddings)
    dataset = Datasets('deep_fashion', num_examples = 5000)
    dataset.clean_dataset()
    dataset.load_dataset()

    # Compute metrics
    ks = [10, 30, 50]
    metrics = []
    for k in ks:
        print('\nComputing metrics for top-' + str(k) + ' recommendations: ')
        metrics_k = Metrics('deep_fashion',  num_examples = 5000, top_k = k, compute_sim = True)
        metrics_k.compute_recall()
        metrics_k.get_recommendations(num_examples = 2)
        print('\n' + '-' * 100)

'''
    TODOs:
    1. Pass args from CLI to run required modules (not all)
'''