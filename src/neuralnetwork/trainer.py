import os
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import trange, tqdm
import time

from src.utils.utils import epoch_time

dataset_paths = {
    'deep_fashion': 'fashion_dataset',
    'test_data': 'dataset'
}


class Trainer:
    def __init__(self, model, dataset_type, train_dataset, val_dataset, test_dataset, batch_size = 32, lr = 1e-3, device = None):
        '''
            @param model: The model to train
            @param train_dataset: The training dataset
            @param val_dataset: The validation dataset
            @param test_dataset: The test dataset
            @param batch_size: The batch size for training and evaluation
            @param lr: Learning rate for the optimizer
            @param device: Device to run the training (e.g., 'cuda' or 'cpu')
        '''
        self.model = model
        self.train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
        self.val_loader = DataLoader(val_dataset, batch_size = batch_size, shuffle = False)
        self.test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle = False)
        
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr = lr)

        self.dataset_type = dataset_type
        self.models_path = os.path.join('../models', dataset_paths[self.dataset_type])
        os.makedirs(self.models_path, exist_ok = True)

        self.predictions_path = os.path.join('../predictions', dataset_paths[self.dataset_type])
        os.makedirs(self.predictions_path, exist_ok = True)

        self.pos_predictions_path = os.path.join(self.predictions_path, 'positive')
        os.makedirs(self.pos_predictions_path, exist_ok = True)

        self.neg_predictions_path = os.path.join(self.predictions_path, 'negative')
        os.makedirs(self.neg_predictions_path, exist_ok = True)
        
        return
        
    def train_one_epoch(self):
        epoch_loss = 0
        epoch_acc = 0

        # Set to train mode
        self.model.train()

        for batch in tqdm(self.train_loader, desc = "Training", leave = False):

            x = batch['embs'].to(self.device)
            y = batch['relevance_scores'].to(self.device)

            self.optimizer.zero_grad()

            y_pred = self.model(x)

            loss = self.criterion(y_pred, y)

            acc = self.calculate_accuracy(y_pred, y)

            loss.backward()

            self.optimizer.step()

            epoch_loss += loss.item()
            epoch_acc += acc.item()

        return epoch_loss / len(self.train_loader), epoch_acc / len(self.train_loader)
    
    def evaluate(self, iterator, precision_k = 3):
        epoch_loss = 0
        epoch_precision = 0

        # Set to eval mode
        self.model.eval()

        with torch.no_grad():

            for batch in tqdm(iterator, desc = "Evaluating", leave = False):

                queries = batch['query_ids']
                relevance_scores = batch['relevance_scores']
                query_embeddings = batch['embs']

                batch_size, num_items, _ = query_embeddings.shape
                predicted_scores = self.model(query_embeddings.view(-1, query_embeddings.shape[-1]))
                num_classes = predicted_scores.shape[-1]
                predicted_scores_per_query = predicted_scores.view(batch_size, num_items, num_classes)
                epoch_precision += self.calculate_precision_k(predicted_scores_per_query, relevance_scores, precision_k)
      
                loss = self.criterion(predicted_scores, relevance_scores.view(-1))
                epoch_loss += loss.item()

        return epoch_loss / len(iterator) , epoch_precision / (len(iterator) * batch_size)
    
    def train(self, num_epochs = 10):
        best_val_loss = float('inf')
        best_val_precision = -1
        for epoch in trange(num_epochs):
            start_time = time.monotonic()

            train_loss, train_acc = self.train_one_epoch()
            val_loss, val_precision = self.evaluate(self.val_loader)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), os.path.join(self.models_path, 'best_model_loss.pt'))
            
            if val_precision > best_val_precision:
                best_val_precision = val_precision
                torch.save(self.model.state_dict(), os.path.join(self.models_path, 'best_model_precision.pt'))

            end_time = time.monotonic()

            epoch_mins, epoch_secs = epoch_time(start_time, end_time)

            print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
            print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
            print(f'\t Val. Loss: {val_loss:.3f} |  Val. Precision: {val_precision * 100:.2f}%')
    
        return
    

    def predict(self):
        iterator = self.test_loader

        # Set to eval mode
        self.model.eval()

        with torch.no_grad():

            for batch in tqdm(iterator, desc = "Testing", leave = False):

                queries = batch['query_ids']
                item_ids = batch['item_ids']
                relevance_scores = batch['relevance_scores']
                query_embeddings = batch['embs']

                batch_size, num_items, _ = query_embeddings.shape
                predicted_scores = self.model(query_embeddings.view(-1, query_embeddings.shape[-1]))
                num_classes = predicted_scores.shape[-1]
                predicted_scores_per_query = predicted_scores.view(batch_size, num_items, num_classes)

                self.save_recommendations(queries, item_ids, predicted_scores_per_query, relevance_scores)

        return

    # TODO: update this
    def test(self):
        test_loss, test_acc = self.evaluate(self.test_loader)
        print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}')
        return
    
    def save_recommendations(self, queries, item_ids, predicted_scores_per_query, relevance_scores):
        batch_size, _, _ = predicted_scores_per_query.shape
        
        # Iterate over each query in the batch
        for i in range(batch_size):
            query_id = queries[i]
            items = item_ids[i]
            predicted_scores_query = predicted_scores_per_query[i]
            relevance_score = relevance_scores[i]

            top_k = torch.topk(predicted_scores_query[:, 1], k = 10)
            top_k_indices = top_k.indices

            gt_index = torch.argmax(relevance_score).item()
            predicted_gt_index = torch.where(top_k_indices == gt_index)[0]

            dest_path = None
            if predicted_gt_index < gt_index:
                dest_path = self.pos_predictions_path
            elif predicted_gt_index >= gt_index:
                dest_path = self.neg_predictions_path

            reordered_items = items[top_k_indices].tolist()
            filename = os.path.join(dest_path, f"{query_id}.txt")
            with open(filename, 'w') as f:
                f.write("\n".join(map(str, reordered_items)))

        return

    def calculate_precision_k(self, predicted_scores_per_query, relevance_scores, precision_k):
        batch_size, _, _ = predicted_scores_per_query.shape
        
        precision = 0
        # Iterate over each query in the batch
        for i in range(batch_size):
            relevance_score = relevance_scores[i]
            predicted_scores_query = predicted_scores_per_query[i]

            # Sorting by class 1 (but if prob of 1 is less than 0.5?)
            top_k = torch.topk(predicted_scores_query[:, 1], k = precision_k)
            top_k_indices = top_k.indices
            
            gt_index = torch.argmax(relevance_score).item()
            precision_at_k = 1.0 if gt_index in top_k_indices else 0.0

            precision += precision_at_k
        
        return precision

    def calculate_accuracy(self, y_pred, y):
        top_pred = y_pred.argmax(1, keepdim = True)
        correct = top_pred.eq(y.view_as(top_pred)).sum()
        acc = correct.float() / y.shape[0]
        return acc
