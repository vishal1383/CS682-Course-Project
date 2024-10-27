from Datasets import Datasets
from Metrics import Metrics

if __name__ == "__main__":
    # Clean & load the dataset (embeddings)
    dataset = Datasets('deep_fashion', num_examples = 100)
    dataset.clean_dataset()
    dataset.load_dataset()

    # Compute metrics
    metrics_10 = Metrics('deep_fashion',  num_examples = 100, top_k = 10, compute_sim = True)
    metrics_30 = Metrics('deep_fashion',  num_examples = 100, top_k = 30, compute_sim = True)
    metrics_50 = Metrics('deep_fashion',  num_examples = 100, top_k = 50, compute_sim = True)

    print('Recall@k for k=10:', metrics_10.compute_recall())
    print('Recall@k for k=30:', metrics_30.compute_recall())
    print('Recall@k for k=50:', metrics_50.compute_recall())

    print('-' * 50)
    print('Printing the misclassified examples for k=10: ')
    print(metrics_10.get_recommendations(num_examples = 2))


'''
    TODOs:
    1. Pass args from CLI to run required modules (not all)
'''