import argparse
import os
import pandas as pd
import random
from PIL import Image

from sklearn.model_selection import train_test_split

from src.utils.plot_utils import PlotUtils
from src.clip.generate_embeddings import GenerateEmbeddings
from src.finetuning.metrics import Metrics

from transformers import CLIPModel, CLIPProcessor
from src.finetuning.clip_finetune import CLIPFinetune

dataset_paths = {
    'deep_fashion': 'fashion_dataset',
    'test_data': 'dataset'
}

def parse_args():
    parser = argparse.ArgumentParser(description = "Train and Evaluate Your Model")
    parser.add_argument('--dataset_type', type = str, default = 'deep_fashion', help = "Dataset type")
    parser.add_argument('--task', type = str, choices = ['finetune', 'inference', 'get_recommendations'], default = 'train')
    parser.add_argument('--raw_data_root_path', type = str, default = '../datasets', help = "Path to raw dataset")
    parser.add_argument('--embeddings_root_path', type = str, default= '../embeddings', help = "Path to embeddings")
    parser.add_argument('--k', type = int, default = 10, help = "Top k retrieved results to consider")
    parser.add_argument('--create_dataset', type = bool, default = True, help = "Should create dataset")
    parser.add_argument('--compute_metrics', type = bool, default = True, help = "Should compute metrics")
    parser.add_argument('--num_examples', type = int, default = 100, help = "Number of examples to be considered in the raw dataset")
    parser.add_argument('--predictions_root_path', type = str, default = '../predictions', help = "Path to save the predictions")
    parser.add_argument('--retrieved_data_root_path', type = str, default = '../retrieved_items', help = "Path to retrieved data")
    return parser.parse_args()

def load_checkpoints(args):
    models_path = os.path.join('../models', dataset_paths[args.dataset_type], 'finetune')
    
    model = CLIPModel.from_pretrained(os.path.join(models_path, "fine_tuned_clip_model"))
    processor = CLIPProcessor.from_pretrained(os.path.join(models_path, "fine_tuned_clip_processor"))
    return model, processor

def load_data(args):
    data_df = pd.read_csv(os.path.join(args.raw_data_root_path, dataset_paths[args.dataset_type], 'dataset.csv'))[['id', 'query']]
    files = data_df['id'].tolist()
    id_query_dict = dict(zip(data_df['id'], data_df['query']))

    seed = 42
    random.seed(seed)
    random.shuffle(files)
    train_files, temp_files = train_test_split(files, train_size = 0.7, random_state = seed)
    val_files, test_files = train_test_split(temp_files, test_size = 0.15 / (0.15 + 0.15), random_state = seed)

    image_paths = []
    texts = []
    for id in train_files:
        texts.append(id_query_dict[id])
        image_paths.append(os.path.join(args.raw_data_root_path, dataset_paths[args.dataset_type], 'rcnn_cropped_images', id, str(id) + '.jpg'))

    return image_paths, texts

def save_embeddings(args, model, processor):
    embbeding_util = GenerateEmbeddings(model, processor)
    data_df = pd.read_csv(os.path.join(args.raw_data_root_path, dataset_paths[args.dataset_type], 'dataset.csv'))
    
    # Ids in the dataset corresponding to image-text pair
    ids = data_df['id'].tolist()
    
    image_filenames = [data_df.iloc[i]['filename'] for i in range(len(data_df))]
    image_paths = [os.path.join(args.raw_data_root_path, dataset_paths[args.dataset_type], 'images', image_name) for image_name in image_filenames]
    texts = data_df['query'].to_list()

    embedding_prefix = os.path.join(args.embeddings_root_path, dataset_paths[args.dataset_type], 'finetune')
    embedding_query_prefix = os.path.join(embedding_prefix, 'queries')
    os.makedirs(embedding_query_prefix, exist_ok = True)
    
    # - embeddings/{dataset}/queries
    embedding_item_prefix = os.path.join(embedding_prefix, 'items')
    os.makedirs(embedding_item_prefix, exist_ok = True)

    # Generate and save embeddings for text-image pairs
    for i in range(len(texts)):
        embs = embbeding_util.generate_embeddings([texts[i]], [Image.open(image_paths[i])])
        embbeding_util.save_embeddings(embs['text_embs'], os.path.join(embedding_query_prefix, str(ids[i]) + '.emb'))
        embbeding_util.save_embeddings(embs['image_embeds'], os.path.join(embedding_item_prefix, str(ids[i]) + '.emb'))

        if i == len(texts) - 1 or (i >= 100 and i % 100 == 0):
            print('|', '-' * 10, 'Done processing ' + str(i + 1) + ' text-image pairs')
    
    print('\n' + '-' * 50)
    return

def finetune(args):
    image_paths, texts = load_data(args)
    
    clipFinetune = CLIPFinetune(dataset_type = args.dataset_type, images = image_paths, texts = texts)
    model, processor = clipFinetune.train()
    
    save_embeddings(args,model, processor)

    metrics = Metrics('deep_fashion',  num_examples = args.num_examples, compute_sim = True)
    metrics.compute_recall()
    return

def run_inference(args):
    model, processor = load_checkpoints(args)
    save_embeddings(args, model, processor)

    metrics = Metrics('deep_fashion',  num_examples = args.num_examples, compute_sim = True)
    metrics.compute_recall()
    metrics.save_predictions()
    return

# Show recommendations after re-ranking vs before re-ranking
# Could show positive or negative examples - use type parameter
def show_recommendations(args, type = 'positive', examples = 10, shuffle = True):
    plotUtils = PlotUtils()
    
    data_df = pd.read_csv(os.path.join(args.raw_data_root_path, dataset_paths[args.dataset_type], 'dataset.csv'))[['id', 'query']]
    id_query_dict = dict(zip(data_df['id'], data_df['query']))
    
    predictions_path = os.path.join(args.predictions_root_path, dataset_paths[args.dataset_type], 'finetune', type)
    file_names = os.listdir(predictions_path)
    if shuffle == True:
        random.shuffle(file_names)
    
    for file_name in file_names[:examples]:
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
        clip_img_path = os.path.join(args.retrieved_data_root_path, dataset_paths[args.dataset_type], '10', query_id + '_c.txt')
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
    args = parse_args()

    if args.task == 'finetune':
        finetune(args)

    if args.task == 'inference':
        run_inference(args)

    if args.task == 'get_recommendations':
        show_recommendations(args)