import torch
from sklearn.model_selection import train_test_split
import argparse
import os

from src.neuralnetwork.model import MLP
from src.neuralnetwork.retrieved_dataset import RetrievedDataset
from src.neuralnetwork.trainer import Trainer
from src.utils.utils import split_dataset

dataset_paths = {
    'deep_fashion': 'fashion_dataset',
    'test_data': 'dataset'
}

def parse_args():
    parser = argparse.ArgumentParser(description = "Train and Evaluate Your Model")
    parser.add_argument('--dataset_type', type = str, default = 'deep_fashion', help = "Dataset type")
    parser.add_argument('--k', type = int, default = 10, help = "Top k retrieved results to consider")
    parser.add_argument('--batch_size', type = int, default = 32, help = "Batch size for training")
    parser.add_argument('--num_epochs', type = int, default = 10, help = "Number of epochs to train")
    parser.add_argument('--learning_rate', type = float, default = 0.00001, help = "Learning rate for the optimizer")
    parser.add_argument('--retrieved_data_root_path', type = str, default = '../retrieved_items', help = "Path to retrieved data")
    parser.add_argument('--embeddings_root_path', type = str, default= '../embeddings', help = "Path to embeddings")
    return parser.parse_args()

def load_data(data_dir_path = '../retrieved_items', embeddings_dir_path = '../embeddings'):

    data_dir_path = os.path.join(data_dir_path)
    files = split_dataset(data_dir_path)
    train_files = files['train_files']
    val_files = files['val_files']
    test_files = files['test_files']

    train_dataset = RetrievedDataset(train_files, embeddings_dir_path, mode = 'train')
    val_dataset = RetrievedDataset(val_files, embeddings_dir_path, mode = 'val')
    test_dataset = RetrievedDataset(test_files, embeddings_dir_path, mode = 'test')
    return train_dataset, val_dataset, test_dataset

def run_training(args):
    data_dir_path = os.path.join(args.retrieved_data_root_path, dataset_paths[args.dataset_type], str(args.k))
    embeddings_dir_path = os.path.join(args.embeddings_root_path, dataset_paths[args.dataset_type])
    
    train_dataset, val_dataset, test_dataset = load_data(data_dir_path, embeddings_dir_path)
    model = MLP(input_dim = 1024, output_dim = 2)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    trainer = Trainer(model, args.dataset_type, train_dataset, val_dataset, test_dataset, args.batch_size, args.learning_rate, device)

    trainer.train(num_epochs = args.num_epochs)

if __name__ == "__main__":
    # Parse arguments
    args = parse_args()
    
    # Run the training process
    run_training(args)
