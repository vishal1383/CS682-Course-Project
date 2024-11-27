import subprocess
import argparse

# Parse arguments for method selection and additional parameters
def parse_args():
    parser = argparse.ArgumentParser(description = "Script to trigger different approaches")
    
    parser.add_argument('--method', type = str, choices = ['clip', 'neuralnetwork', 'finetuning', 'all'], required = True, help = "Choose method: 'clip' for running the basic clip model and so on")
    parser.add_argument('--dataset_type', type = str, default = 'deep_fashion', help = "Dataset type")
    parser.add_argument('--k', type = int, default = 10, help = "Top k retrieved results to consider")
    
    # Arguments for the clip method
    parser.add_argument('--create_dataset', type = bool, default = True, help = "Should create dataset")
    parser.add_argument('--compute_metrics', type = bool, default = True, help = "Should compute metrics")
    parser.add_argument('--num_examples', type = int, default = 5000, help = "Number of examples to be considered in the raw dataset")
    
    # Arguments for the neural network method
    parser.add_argument('--batch_size', type = int, default = 32, help = "Batch size for training")
    parser.add_argument('--num_epochs', type = int, default = 10, help = "Number of epochs to train")
    parser.add_argument('--learning_rate', type = float, default = 0.00001, help = "Learning rate for the optimizer")
    parser.add_argument('--retrieved_data_root_path', type = str, default = '../retrieved_items', help = "Path to retrieved data")
    parser.add_argument('--embeddings_root_path', type = str, default= '../embeddings', help = "Path to embeddings")
    return parser.parse_args()

# Run CLIP
def run_clip(args):
    print("Running CLIP...")
    command = [
        'python', 'clip/main.py', 
        '--dataset_type', args.dataset_type,
        '--compute_metrics', str(args.compute_metrics),
        '--num_examples', str(args.num_examples)
    ]
    subprocess.run(command, check = True)

# Run neural network
def run_neuralnetwork(args):
    print("Running neural network...")
    
    command = [
        'python', 'neuralnetwork/main.py', 
        '--dataset_type', args.dataset_type,
        '--batch_size', str(args.batch_size),
        '--num_epochs', str(args.num_epochs),
        '--learning_rate', str(args.learning_rate),
        '--retrieved_data_root_path', args.retrieved_data_root_path,
        '--embeddings_root_path', args.embeddings_root_path
    ]
    subprocess.run(command, check = True)

if __name__ == "__main__":
    # Parse the command line arguments
    args = parse_args()

    # Trigger appropriate method based on input
    if args.method == 'clip':
        run_clip(args)
    
    elif args.method == 'neuralnetwork':
        run_neuralnetwork(args)
    
    if args.method == 'all':
        run_clip(args)
        run_neuralnetwork(args)

