import os
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm.notebook import trange, tqdm
import time
from utils.utils import epoch_time

class Trainer:
    def __init__(self, model, train_dataset, val_dataset, test_dataset, batch_size = 32, lr = 1e-3, device = None):
        """
        Initialize the Trainer class.
        
        :param model: The model to train
        :param train_dataset: The training dataset
        :param val_dataset: The validation dataset
        :param test_dataset: The test dataset
        :param batch_size: The batch size for training and evaluation
        :param lr: Learning rate for the optimizer
        :param device: Device to run the training (e.g., 'cuda' or 'cpu')
        """
        self.model = model
        self.train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
        self.val_loader = DataLoader(val_dataset, batch_size = batch_size, shuffle = False)
        self.test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle = False)
        
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr = lr)
        
    def train_one_epoch(self):
        epoch_loss = 0
        epoch_acc = 0

        # Set to train mode
        self.model.train()

        for batch in tqdm(self.train_loader, desc = "Training", leave = False):

            x = batch['emb'].to(self.device)
            y = batch['label'].to(self.device)

            self.optimizer.zero_grad()

            y_pred = self.model(x)

            y = torch.nonzero(y, as_tuple=False)
            y = y[:, 1]

            loss = self.criterion(y_pred, y)

            acc = self.calculate_accuracy(y_pred, y)

            loss.backward()

            self.optimizer.step()

            epoch_loss += loss.item()
            epoch_acc += acc.item()

        return epoch_loss / len(self.train_loader), epoch_acc / len(self.train_loader)

    def evaluate(self, iterator):
        epoch_loss = 0
        epoch_acc = 0

        self.model.eval()

        with torch.no_grad():

            for batch in tqdm(iterator, desc = "Evaluating", leave=False):

                x = batch['emb'].to(self.device)
                y = batch['label'].to(self.device)

                y_pred = self.model(x)

                y = torch.nonzero(y, as_tuple=False)
                y = y[:, 1]

                loss = self.criterion(y_pred, y)

                acc = self.calculate_accuracy(y_pred, y)

                epoch_loss += loss.item()
                epoch_acc += acc.item()

        return epoch_loss / len(iterator), epoch_acc / len(iterator)
    
    def train(self, num_epochs = 10):
        best_valid_loss = float('inf')
        for epoch in trange(num_epochs):
            start_time = time.monotonic()

            train_loss, train_acc = self.train_one_epoch()
            valid_loss, valid_acc = self.evaluate(self.val_loader)

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(self.model.state_dict(), 'best_model.pt')

            end_time = time.monotonic()

            epoch_mins, epoch_secs = epoch_time(start_time, end_time)

            print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
            print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
            print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')
    
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

                y = torch.nonzero(y, as_tuple = False)
                y = y[:, 1]

                ids.append(batch['query_id'])
                labels.append(y.cpu())
                probs.append(y_prob.cpu())

        ids = torch.cat(ids, dim = 0)
        labels = torch.cat(labels, dim = 0)
        probs = torch.cat(probs, dim = 0)

        return ids, labels, probs
    
    def test(self):
        test_loss, test_acc = self.evaluate(self.test_loader)
        print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}')
