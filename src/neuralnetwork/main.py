import torch
from model import MLP
from sklearn.model_selection import train_test_split
from src.neuralnetwork.retrieved_dataset import RetrievedDataset
from trainer import Trainer
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description = "Train and Evaluate Your Model")
    parser.add_argument('--batch_size', type= int, default = 32, help= "Batch size for training")
    parser.add_argument('--num_epochs', type= int, default = 10, help= "Number of epochs to train")
    parser.add_argument('--learning_rate', type= float, default = 0.001, help= "Learning rate for the optimizer")
    parser.add_argument('--train_data_dir', type= str, default= 'data/train_data', help= "Path to training data")
    parser.add_argument('--test_data_dir', type= str, default= 'data/test_data', help= "Path to test data")
    return parser.parse_args()

def load_data(data_dir_path = '../retrieved_items', embeddings_dir_path = 'embeddings/test_dataset'):
    dataset = RetrievedDataset(data_dir_path, embeddings_dir_path)

    # Split the dataset into training, validation, and testing sets
    train_dataset, temp_dataset = train_test_split(dataset, test_size = 0.3, random_state = 42)
    val_dataset, test_dataset = train_test_split(temp_dataset, test_size = 0.5, random_state = 42)
    return train_dataset, val_dataset, test_dataset

def run_training(args):
    train_dataset, val_dataset, test_dataset = load_data(args.train_data_dir, args.test_data_dir, args.batch_size)
    model = MLP()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    trainer = Trainer(model, train_dataset, val_dataset, test_dataset, args.batch_size, args.learning_rate, device)

    trainer.train()


if __name__ == "__main__":
    # Parse arguments
    args = parse_args()
    
    # Run the training process
    run_training(args)
