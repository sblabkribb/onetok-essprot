import torch, time, os, random
import torch.nn as nn
import pandas as pd
import numpy as np
import sklearn.metrics as met
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader



num_workers = min(os.cpu_count(), 32)
scaler = torch.cuda.amp.GradScaler()


# set seed of dataloader
g = torch.Generator()
g.manual_seed(42)
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    random.seed(worker_seed)



def set_dataloader(X_all, y_all, batch_size, device):
    # split to train & validation dataset
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_all, y_all, test_size=0.1, stratify=y_all, random_state=42
    )
    
    # calculate class weight
    class_counts = np.bincount(y_train.astype(int))
    pos_weight = torch.tensor(class_counts[0] / class_counts[1], dtype=torch.float).to(device)
    print(f"Number by class: {class_counts} | Positive weight: {pos_weight}")
    
    # convert dataset to tensor
    X_train = torch.FloatTensor(X_train)
    X_valid = torch.FloatTensor(X_valid)
    y_train = torch.FloatTensor(y_train)
    y_valid = torch.FloatTensor(y_valid)        
    print(f"X_train: {X_train.shape}, y_train: {y_train.shape} |",
          f"X_val: {X_valid.shape}, y_val: {y_valid.shape}")
    
    # generate dataloader
    train_dataset = TensorDataset(X_train, y_train)
    valid_dataset = TensorDataset(X_valid, y_valid)            
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size,
        shuffle=True, worker_init_fn=seed_worker, generator=g,
        pin_memory=True, num_workers=num_workers, persistent_workers=True, prefetch_factor=8
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=batch_size,
        pin_memory=True, num_workers=num_workers, persistent_workers=True, prefetch_factor=8
    )
    print(f"Training batches: {len(train_loader)}, Valid batches: {len(valid_loader)}")

    return train_loader, valid_loader, pos_weight



def train_model(model, train_loader, valid_loader, criterion, optimizer,
                model_path, model_name, num_epochs=1000, patience=30, min_delta=1e-4):
    # initialize variables
    history = {'epoch': [],
               'train_loss': [],
               'train_mcc': [],
               'valid_loss': [],
               'valid_mcc': []}
    best_val_mcc = -1.
    counter = 0

    for epoch in range(1, num_epochs + 1):
        time_epoch = time.time()
        
        # model training
        model.train()
        train_loss = 0.0
        batch_len = 0
        train_logits = []
        train_labels = []

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(model.device, non_blocking=True)
            y_batch = y_batch.to(model.device, non_blocking=True)

            # prediction with mixed precision
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                logits = model(X_batch)
                loss = criterion(logits, y_batch)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            # gather results
            train_logits.append(logits.detach())
            train_labels.append(y_batch)
            train_loss += loss.item()
            batch_len += 1

        # calculate metrics        
        train_loss = train_loss / batch_len
        train_logits = torch.cat(train_logits).detach().to('cpu', non_blocking=True)
        train_labels = torch.cat(train_labels).to('cpu', non_blocking=True)
        train_preds = torch.round(torch.sigmoid(train_logits))
        train_mcc = met.matthews_corrcoef(train_labels.numpy(), train_preds.numpy())

        # model validation
        model.eval()
        valid_loss = 0.0
        batch_len = 0
        valid_logits = []
        valid_labels = []
        
        with torch.no_grad():
            for X_batch, y_batch in valid_loader:
                X_batch = X_batch.to(model.device, non_blocking=True)
                y_batch = y_batch.to(model.device, non_blocking=True)
                # prediction with mixed precision
                with torch.cuda.amp.autocast():
                    logits = model(X_batch)
                    loss = criterion(logits, y_batch)
                # gather results
                valid_logits.append(logits.detach())
                valid_labels.append(y_batch)
                valid_loss += loss.item()
                batch_len += 1

        # calculate metrics
        valid_loss = valid_loss / batch_len
        valid_logits = torch.cat(valid_logits).detach().to('cpu', non_blocking=True)
        valid_labels = torch.cat(valid_labels).to('cpu', non_blocking=True)
        valid_preds = torch.round(torch.sigmoid(valid_logits))
        valid_mcc = met.matthews_corrcoef(valid_labels.numpy(), valid_preds.numpy())

        # save history
        history['epoch'].append(epoch)
        history['train_loss'].append(train_loss)
        history['train_mcc'].append(train_mcc)
        history['valid_loss'].append(valid_loss)
        history['valid_mcc'].append(valid_mcc)

        print(f"- Epoch [{epoch}/{num_epochs}] Time: {time.time() - time_epoch:.1f} sec |",
              f"Train loss: {train_loss:.4f}, Train MCC: {train_mcc:.4f} |",
              f"Val loss: {valid_loss:.4f}, Val MCC: {valid_mcc:.4f}")

        # check early stopping
        if  valid_mcc - best_val_mcc > min_delta:
            best_val_mcc = valid_mcc
            counter = 0
            # save model
            torch.save(model._orig_mod.state_dict(), os.path.join(model_path, f'{model_name}.pt'))
        else:
            counter += 1
            if counter >= patience:
                print(f"!!! Early stopping at epoch {epoch} !!!")
                break
    
    return history



def calculate_metrics(embed_ver:str, integ_ver:str, comb_ver:str, y_true, y_prob, y_pred):
    comb_n = len(comb_ver.split('_'))
    
    tn, fp, fn, tp = met.confusion_matrix(y_true, y_pred).ravel()
    model_eval = pd.DataFrame({
        "llm": [embed_ver],
        "integ": [integ_ver],
        "embed": [comb_ver],
        "comb_n": [comb_n],
        "tp": [tp],
        "fp": [fp],
        "tn": [tn],
        "fn": [fn],
        "mcc": [met.matthews_corrcoef(y_true, y_pred)],
        "acc": [met.accuracy_score(y_true, y_pred)],
        "f1": [met.f1_score(y_true, y_pred)],
        "prc": [met.precision_score(y_true, y_pred)],
        "rec": [met.recall_score(y_true, y_pred)],
        "npv": [met.precision_score(1 - y_true, 1 - y_pred)],
        "tnr": [met.recall_score(1 - y_true, 1 - y_pred)],
        "auc-roc": [met.roc_auc_score(y_true, y_prob)],
        "auc-pr": [met.average_precision_score(y_true, y_prob)]
    })
    return model_eval



def test_model(model, test_loader, embed_ver:str, integ_ver:str, comb_ver:str, info_df):
    test_logits = []
    test_labels = []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(model.device, non_blocking=True)
            y_batch = y_batch.to(model.device, non_blocking=True)
            # prediction
            logits = model(X_batch)
            # gather results
            test_logits.append(logits.detach())
            test_labels.append(y_batch)
    
     # calculate metrics
    test_logits = torch.cat(test_logits).detach().to('cpu', non_blocking=True)
    test_labels = torch.cat(test_labels).to('cpu', non_blocking=True)
    test_probs = torch.sigmoid(test_logits)
    test_preds = torch.round(test_probs)
    model_eval = calculate_metrics(
        embed_ver, integ_ver, comb_ver, test_labels.numpy(), test_probs.numpy(), test_preds.numpy()
    )
    # concatenate predictions to inforamations
    df_probs = pd.DataFrame({'conf': test_probs})
    model_prob = pd.concat([info_df, df_probs], axis=1)
    
    return model_eval, model_prob       
            


class Classifier(nn.Module):
    def __init__(self, input_size, num_layers=2):
        super(Classifier, self).__init__()
        layers = [nn.BatchNorm1d(input_size), nn.Dropout(0.5)]
        in_dim = input_size
        out_dim = 256
        for _ in range(num_layers):
            layers.append(nn.Linear(in_dim, out_dim))
            self.initialize_weights(layers[-1])
            layers.append(nn.GELU())
            in_dim = out_dim
            out_dim = max(2, out_dim // 4)
        layers.append(nn.Linear(in_dim, 1))
        self.cls = nn.Sequential(*layers)
        
    def initialize_weights(self, layer):
        nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='linear')
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)
    
    def to(self, device):
        self.device = device
        return super().to(device)
    
    def forward(self, x):
        return self.cls(x).squeeze()