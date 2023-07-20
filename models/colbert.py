import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import BertModel, get_cosine_schedule_with_warmup
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from tqdm import tqdm
import numpy as np
from transformers import AutoTokenizer

MAX_LENGTH = 100


class ColBERTDataset(Dataset):
    def __init__(self, data, tokenizer, device, max_length=MAX_LENGTH):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.device = device

    def __len__(self):
        return len(self.data['label'])

    def __getitem__(self, idx):
        text = self.data['full_text'][idx]
        sentences = self.data['sentences'][idx]
        label = self.data['label'][idx]
        encoded_full_text = self.tokenizer(text,
                                           max_length=self.max_length,
                                           add_special_tokens=True,
                                           padding='max_length',
                                           truncation=True,
                                           return_attention_mask=True,
                                           return_token_type_ids=True,
                                           return_tensors='pt')
        input_ids_full_text = encoded_full_text['input_ids'].squeeze(0).to(self.device)
        attention_mask_full_text = encoded_full_text['attention_mask'].squeeze(0).to(self.device)
        token_type_ids_full_text = encoded_full_text['token_type_ids'].squeeze(0).to(self.device)
        encoded_sent = self.tokenizer(sentences,
                                      max_length=self.max_length,
                                      add_special_tokens=True,
                                      padding='max_length',
                                      truncation=True,
                                      return_attention_mask=True,
                                      return_token_type_ids=True,
                                      return_tensors='pt')
        input_ids_sent = encoded_sent['input_ids'].squeeze(0).to(self.device)
        attention_mask_sent = encoded_sent['attention_mask'].squeeze(0).to(self.device)
        token_type_ids_sent = encoded_sent['token_type_ids'].squeeze(0).to(self.device)
        return {'input_full': {'input_ids': input_ids_full_text,
                               'attention_mask': attention_mask_full_text,
                               'token_type_ids': token_type_ids_full_text},
                'input_sent': {'input_ids': input_ids_sent,
                               'attention_mask': attention_mask_sent,
                               'token_type_ids': token_type_ids_sent},
                'label': label}


class ColBERT(nn.Module):
    def __init__(self, n_sentences, freeze_BERT=True):
        super(ColBERT, self).__init__()
        self.bert = BertModel.from_pretrained("DeepPavlov/rubert-base-cased-conversational")
        self.n_sentences = n_sentences

        # Freeze the BERT layers
        if freeze_BERT:
            for param in self.bert.parameters():
                param.requires_grad = False

        self.hidden_layers_sent = nn.ModuleList([
            nn.Sequential(
                nn.Linear(768, 32),
                nn.Dropout(),
                nn.ReLU(),
                nn.Linear(32, 8),
                nn.ReLU()
            ) for _ in range(n_sentences)
        ])

        self.hidden_layers_full = nn.Sequential(
            nn.Linear(768, 256),
            nn.Dropout(),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU()
        )

        self.concat_layers = nn.Sequential(
            nn.Linear(104, 512),
            nn.Dropout(),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 2)
        )

    def forward(self, input_sent, input_full):
        batch_size = input_sent['attention_mask'].shape[0]
        sent_features = []
        for i in range(self.n_sentences):
            input = {k: v[:, i, :] for k, v in input_sent.items()}
            output_bert = self.bert(**input).pooler_output
            sent_features_i = self.hidden_layers_sent[i](output_bert)
            sent_features.append(sent_features_i)

        output_bert = self.bert(**input_full)
        full_features = self.hidden_layers_full(output_bert.pooler_output)

        sent_features = torch.stack(sent_features, dim=1)
        concat_features = torch.cat([sent_features.view(batch_size, -1), full_features.view(batch_size, -1)], dim=1)
        output = self.concat_layers(concat_features)
        return output


def create_dataset(split_dataset, tokenizer, device):
    dataset = {
        'full_text': [' '.join(s[0]) for s in split_dataset],
        'sentences': [s[0] for s in split_dataset],
        'label': [s[1] for s in split_dataset],
    }
    tokenized_dataset = ColBERTDataset(dataset, tokenizer, device)
    tokenized_dataset_test, tokenized_dataset_train = random_split(
        tokenized_dataset, lengths=[0.2, 0.8])
    return {'train': tokenized_dataset_train, 'test': tokenized_dataset_test}


def compute_metrics(predictions, labels):
    pr, rec, fs, _ = precision_recall_fscore_support(labels, predictions, average='binary')
    return {'precision': pr, 'recall': rec, 'fscore': fs, 'accuracy': accuracy_score(labels, predictions)}


def evaluate(testloader, model, criterion, device):
    eval_logits = []
    eval_labels = []
    eval_loss = 0.0
    with torch.no_grad():
        for data in tqdm(testloader):
            labels = data.get("label")
            eval_labels.extend(labels)
            labels = labels.to(device)
            logits = model(input_sent=data.get('input_sent'), input_full=data.get('input_full'))
            loss = criterion(logits, labels)
            eval_loss += loss.item()
            _, predicted = torch.max(logits.data, 1)
            eval_logits.extend(predicted)
    metrics = compute_metrics(torch.stack(eval_logits).cpu(), np.stack(eval_labels))
    metrics['loss'] = eval_loss / len(testloader)
    return metrics


def load_model(device, freeze_bert=True):
    ColBERT_model = ColBERT(5, freeze_bert)
    ColBERT_model = ColBERT_model.to(device)
    tokenizer = AutoTokenizer.from_pretrained("DeepPavlov/rubert-base-cased-conversational")
    return ColBERT_model, tokenizer


def train_epoch(model, trainloader, testloader, n_batches, epoch, device, optimizer, criterion, scheduler, writer,
                eval_step, models_path, best_f1):
    model.train()
    running_loss = 0.0
    train_logits = []
    train_labels = []
    for i, data in tqdm(enumerate(trainloader, 0), total=n_batches):
        global_step = epoch * n_batches + i

        # get the inputs; data is a list of [inputs, labels]
        labels = data.get("label").to(device)
        train_labels.extend(labels)
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(input_sent=data.get('input_sent'), input_full=data.get('input_full'))
        _, predicted = torch.max(outputs.data, 1)
        train_logits.extend(predicted)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()

        # print statistics
        running_loss += loss.item()
        writer.add_scalar("learning_rate", scheduler.get_lr()[0], global_step)
        if (i + 1) % eval_step == 0:
            train_loss = running_loss / n_batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {train_loss:.3f} lr: {scheduler.get_lr()[0]:.5f}')

            model.eval()
            metrics = evaluate(testloader, model, criterion, device)
            for m, value in metrics.items():
                writer.add_scalar(f"{m}/test", value, global_step)
            print('\t'.join([f'{k.upper()}: {v:.2f}' for k, v in metrics.items()]))
            if metrics['fscore'] > best_f1:
                torch.save(model.state_dict(), os.path.join(models_path, f'checkpoint_{global_step}'))
                best_f1 = metrics['fscore']

            writer.add_scalar("loss/train", train_loss, global_step)
            writer.flush()
            running_loss = 0.0
    train_metrics = compute_metrics(torch.stack(train_logits).cpu(), torch.stack(train_labels).cpu())
    for m, value in train_metrics.items():
        writer.add_scalar(f"{m}/train", value, global_step)
    print('\t'.join([f'{k.upper()}: {v:.2f}' for k, v in train_metrics.items()]))


def train(dataset, model, device, models_path, epochs=5, wr=0.3, bs=16, wd=0.1, lr=5e-5, save_ratio=4):
    PATH = os.path.join(models_path, f'colbert_no_pretrain_wr{wr}_bs{bs}_wd{wd}_lr{lr}')
    writer = SummaryWriter(log_dir=PATH)
    trainloader = DataLoader(dataset['train'], batch_size=bs, shuffle=True)
    testloader = DataLoader(dataset['test'], batch_size=bs, shuffle=False)

    n_batches = len(trainloader)
    eval_step = len(dataset['train']) // bs // save_ratio

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=n_batches * epochs * wr,
        num_training_steps=n_batches * epochs
    )
    best_f1 = 0
    for epoch in range(epochs):
        train_epoch(model, trainloader, testloader, n_batches, epoch, device, optimizer, criterion, scheduler, writer,
                    eval_step, models_path, best_f1)
    writer.close()
