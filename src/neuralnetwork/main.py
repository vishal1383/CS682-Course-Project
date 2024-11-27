import torch
import argparse
import os
from PIL import Image
import pandas as pd
import random

from src.neuralnetwork.model import MLP
from src.neuralnetwork.retrieved_dataset import RetrievedDataset
from src.neuralnetwork.trainer import Trainer
from src.utils.utils import split_dataset
from src.utils.plot_utils import PlotUtils

dataset_paths = {
    'deep_fashion': 'fashion_dataset',
    'test_data': 'dataset'
}

def parse_args():
    parser = argparse.ArgumentParser(description = "Train and Evaluate Your Model")
    parser.add_argument('--dataset_type', type = str, default = 'deep_fashion', help = "Dataset type")
    parser.add_argument('--k', type = int, default = 10, help = "Top k retrieved results to consider")
    parser.add_argument('--batch_size', type = int, default = 32, help = "Batch size for training")
    parser.add_argument('--task', type = str, choices = ['train', 'inference', 'get_recommendations'], default = 'train')
    parser.add_argument('--num_epochs', type = int, default = 10, help = "Number of epochs to train")
    parser.add_argument('--learning_rate', type = float, default = 0.0001, help = "Learning rate for the optimizer")
    parser.add_argument('--raw_data_root_path', type = str, default = '../datasets', help = "Path to raw dataset")
    parser.add_argument('--retrieved_data_root_path', type = str, default = '../retrieved_items', help = "Path to retrieved data")
    parser.add_argument('--embeddings_root_path', type = str, default= '../embeddings', help = "Path to embeddings")
    parser.add_argument('--model_root_path', type = str, default = '../models', help = "Path to save model")
    parser.add_argument('--predictions_root_path', type = str, default = '../predictions', help = "Path to save the predictions")
    return parser.parse_args()

def load_data(data_dir_path, embeddings_dir_path):

    data_dir_path = os.path.join(data_dir_path)
    files = split_dataset(data_dir_path)
    train_files = files['train_files']
    val_files = files['val_files']
    test_files = files['test_files']

    train_dataset = RetrievedDataset(train_files, embeddings_dir_path, mode = 'train')
    val_dataset = RetrievedDataset(val_files, embeddings_dir_path, mode = 'val')
    test_dataset = RetrievedDataset(test_files, embeddings_dir_path, mode = 'test')

    print('Dataset stats')
    print('Len of train dataset', len(train_dataset))
    print('Len of val dataset', len(val_dataset))
    print('Len of test dataset', len(test_dataset))
    
    return train_dataset, val_dataset, test_dataset

def run_model(args):
    model = MLP(input_dim = 1024, output_dim = 2)
    
    data_dir_path = os.path.join(args.retrieved_data_root_path, dataset_paths[args.dataset_type], str(args.k))
    embeddings_dir_path = os.path.join(args.embeddings_root_path, dataset_paths[args.dataset_type])
    train_dataset, val_dataset, test_dataset = load_data(data_dir_path, embeddings_dir_path)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    trainer = Trainer(model, args.dataset_type, train_dataset, val_dataset, test_dataset, args.batch_size, args.learning_rate, device)

    trainer.train(num_epochs = args.num_epochs)
    trainer.predict()
    show_recommendations(args)
    return

def run_inference(args):
    model = MLP(input_dim = 1024, output_dim = 2)

    # Load the saved state_dict
    model.load_state_dict(torch.load(os.path.join(args.model_root_path, dataset_paths[args.dataset_type], 'best_model_loss.pt')))
    
    data_dir_path = os.path.join(args.retrieved_data_root_path, dataset_paths[args.dataset_type], str(args.k))
    embeddings_dir_path = os.path.join(args.embeddings_root_path, dataset_paths[args.dataset_type])
    train_dataset, val_dataset, test_dataset = load_data(data_dir_path, embeddings_dir_path)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    trainer = Trainer(model, args.dataset_type, train_dataset, val_dataset, test_dataset, args.batch_size, args.learning_rate, device)
    trainer.predict()
    show_recommendations(args)
    return

# Show recommendations after re-ranking vs before re-ranking
# Could show positive or negative examples - use type parameter
def show_recommendations(args, type = 'positive', shuffle = True):
    plotUtils = PlotUtils()
    
    data_df = pd.read_csv(os.path.join(args.raw_data_root_path, dataset_paths[args.dataset_type], 'dataset.csv'))[['id', 'query']]
    id_query_dict = dict(zip(data_df['id'], data_df['query']))
    
    predictions_path = os.path.join(args.predictions_root_path, dataset_paths[args.dataset_type], type)
    file_names = os.listdir(predictions_path)
    if shuffle == True:
        random.shuffle(file_names)
    
    for file_name in file_names:
        query_id = file_name.split('.')[0]
        text = id_query_dict[int(query_id)]
        
        # Plot recommendations after re-ranking
        nn_file_path = os.path.join(predictions_path, file_name)
        with open(nn_file_path, 'r') as file:
            nn_recommended_images = {'ids': [], 'imgs': []}
            
            labels = []
            for line in file.readlines():
                img_id = line.strip()
                img_path = os.path.join(args.raw_data_root_path, dataset_paths[args.dataset_type], 'images', img_id + '.jpg')
                img = Image.open(img_path)

                if img_id == query_id:
                    labels.append(True)
                else:
                    labels.append(False)
                
                nn_recommended_images['ids'].append(img_id)
                nn_recommended_images['imgs'].append(img)
            
            plotUtils.plot_recommendations(text + ' (After re-ranking)', nn_recommended_images, labels)

        # Plot recommendations before re-ranking
        # TODO: Change '10' here
        clip_img_path = os.path.join(args.retrieved_data_root_path, dataset_paths[args.dataset_type], '10', file_name)
        with open(clip_img_path, 'r') as file:
            clip_recommended_images = {'ids': [], 'imgs': []}
            
            labels = []
            for line in file.readlines():
                img_id = line.strip()
                img_path = os.path.join(args.raw_data_root_path, dataset_paths[args.dataset_type], 'images', img_id + '.jpg')
                img = Image.open(img_path)

                if img_id == query_id:
                    labels.append(True)
                else:
                    labels.append(False)
                
                clip_recommended_images['ids'].append(img_id)
                clip_recommended_images['imgs'].append(img)
            
            plotUtils.plot_recommendations(text + ' (Before re-ranking)', clip_recommended_images, labels)

    return


if __name__ == "__main__":
    # Parse arguments
    args = parse_args()

     # Trigger appropriate task based on input
    if args.task == 'train':
        run_model(args)
    
    elif args.task == 'inference':
        run_inference(args)
    
    if args.task == 'get_recommendations':
        show_recommendations(args)
