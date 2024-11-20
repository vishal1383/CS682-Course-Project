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
                
                # Iterate over each query in the batch
                for i in range(batch_size):
                    relevance_score = relevance_scores[i]
                    predicted_scores_query = predicted_scores_per_query[i]

                    top_k = torch.topk(predicted_scores_query[:, 1], k = precision_k)
                    top_k_indices = top_k.indices
                    
                    gt_index = torch.argmax(relevance_score).item()
                    precision_at_k = 1.0 if gt_index in top_k_indices else 0.0

                    epoch_precision += precision_at_k
      
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
    
    # TODO: Update this
    def predict(self, iterator):
        # Set to eval mode
        self.model.eval()

        ids = []
        labels = []
        probs = []
        with torch.no_grad():

            for batch in iterator:

                x = batch['emb']
                y = batch['label']

                x = x.to(self.device)

                y_pred = self.model(x)

                y_prob = F.softmax(y_pred, dim = -1)

                ids.append(batch['query_id'])
                labels.append(y.cpu())
                probs.append(y_prob.cpu())

        ids = torch.cat(ids, dim = 0)
        labels = torch.cat(labels, dim = 0)
        probs = torch.cat(probs, dim = 0)

        return ids, labels, probs
    
    # TODO: update this
    def test(self):
        test_loss, test_acc = self.evaluate(self.test_loader)
        print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}')
        return

    def calculate_accuracy(self, y_pred, y):
        top_pred = y_pred.argmax(1, keepdim = True)
        correct = top_pred.eq(y.view_as(top_pred)).sum()
        acc = correct.float() / y.shape[0]
        return acc
