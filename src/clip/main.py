import argparse

from src.clip.raw_dataset import RawDataset
from src.clip.metrics import Metrics
from src.utils.plot_utils import PlotUtils

def parse_args():
    parser = argparse.ArgumentParser(description = "Train and Evaluate Your Model")
    parser.add_argument('--dataset_type', type = str, default = 'deep_fashion', help = "Dataset type")
    parser.add_argument('--k', type = int, default = 10, help = "Top k retrieved results to consider")
    parser.add_argument('--create_dataset', type = bool, default = False, help = "Should create dataset")
    parser.add_argument('--compute_metrics', type = bool, default = True, help = "Should compute metrics")
    parser.add_argument('--num_examples', type = int, default = 100, help = "Number of examples to be considered in the raw dataset")
    return parser.parse_args()

# Creates data and embeddings files
def create_dataset(args):
    # Clean & load the dataset (embeddings)
    dataset = RawDataset(args.dataset_type, num_examples = args.num_examples)
    dataset.clean_dataset()
    dataset.load_dataset()
    return

# Gives the first set of retrieved results
def compute_metrics(args):
    ks = [10]
    metrics = []
    for k in ks:
        print('\nComputing metrics for top-' + str(k) + ' recommendations: ')
        metrics_k = Metrics(args.dataset_type,  num_examples = args.num_examples, top_k = k, compute_sim = True)
        metrics.append(metrics_k.compute_recall())
        # metrics_k.save_recommendations()
        # metrics_k.get_recommendations(num_examples = 2)
        print('\n' + '-' * 100)
    
    plotUtils = PlotUtils()
    plotUtils.plot_metric(metrics, ks)

if __name__ == "__main__":
    args = parse_args()
    print(args)

    if args.create_dataset == True:
        create_dataset(args)

    if args.compute_metrics == True:
        compute_metrics(args)
    

'''
    TODOs:
    1. Pass args from CLI to run required modules (not all)
'''