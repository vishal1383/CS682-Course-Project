import torch
from sklearn.model_selection import train_test_split
import argparse

from src.neuralnetwork.model import MLP
from src.neuralnetwork.retrieved_dataset import RetrievedDataset
from src.neuralnetwork.trainer import Trainer

def parse_args():
    parser = argparse.ArgumentParser(description = "Train and Evaluate Your Model")
    parser.add_argument('--batch_size', type= int, default = 32, help= "Batch size for training")
    parser.add_argument('--num_epochs', type= int, default = 10, help= "Number of epochs to train")
    parser.add_argument('--learning_rate', type= float, default = 0.001, help= "Learning rate for the optimizer")
    parser.add_argument('--data_dir_path', type= str, default= '../retrieved_items', help= "Path to retrieved data")
    parser.add_argument('--embeddings_dir_path', type= str, default= '../embeddings/test_dataset', help= "Path to embeddings")
    return parser.parse_args()

def load_data(data_dir_path = '../retrieved_items', embeddings_dir_path = '../embeddings/test_dataset'):
    dataset = RetrievedDataset(data_dir_path, embeddings_dir_path)
    print(len(dataset))

    # Split the dataset into training, validation, and testing sets
    train_dataset, temp_dataset = train_test_split(dataset, test_size = 0.3, random_state = 42)
    print(len(train_dataset))
    val_dataset, test_dataset = train_test_split(temp_dataset, test_size = 0.5, random_state = 42)
    return train_dataset, val_dataset, test_dataset

def run_training(args):
    train_dataset, val_dataset, test_dataset = load_data(args.data_dir_path, args.embeddings_dir_path)
    model = MLP()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    trainer = Trainer(model, train_dataset, val_dataset, test_dataset, args.batch_size, args.learning_rate, device)

    trainer.train()


if __name__ == "__main__":
    # Parse arguments
    args = parse_args()
    
    # Run the training process
    run_training(args)
