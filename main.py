from Datasets import Datasets
from Metrics import Metrics

if __name__ == "__main__":
    # Clean & load the dataset (embeddings)
    dataset = Datasets('deep_fashion', num_examples = 5000)
    dataset.clean_dataset()
    dataset.load_dataset()

    # Compute metrics
    metrics = Metrics('deep_fashion',  num_examples = 5000, compute_sim = True)

    print('Recall:', metrics.compute_recall())

    print('-' * 50)
    print('Printing the misclassified examples: ')
    print(metrics.get_misclassifications())