import os
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification, AutoModel
from transformers import TrainingArguments, Trainer
import jsonlines
import torch
from functools import partial
from torch.utils.data import Dataset, DataLoader, random_split


from sklearn.metrics import precision_recall_fscore_support, accuracy_score

MODELS_NAMES = {
    'bert': 'bert-base-cased',
    'bert_multilingual': 'bert-base-multilingual-cased',
    'rubert': '"DeepPavlov/rubert-base-cased-conversational"'
}

def tokenize_function_with_context(examples, tokenizer):
    return tokenizer(examples['context'], examples['text'], truncation=True, max_length=512, padding='max_length',
                     return_offsets_mapping=True)
def create_dataset(split_dataset, tokenizer, device):
    dataset = {
        'context': [' '.join(s[0][:-1]) for s in split_dataset],
        'text': [s[0][-1] for s in split_dataset],
        'label': [s[1] for s in split_dataset],
    }
    dataset = Dataset.from_dict(dataset)
    dataset.set_format(type='pytorch', device=device)
    tokenized_dataset = dataset.map(partial(tokenize_function_with_context, tokenizer=tokenizer))
    return tokenized_dataset



def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    pr, rec, fs, _ = precision_recall_fscore_support(labels, predictions, average='binary')
    return {'precision': pr, 'recall': rec, 'fscore': fs, 'accuracy': accuracy_score(labels, predictions)}


def load_model(model_name, device, with_classification_head=True):
    model_name = MODELS_NAMES.get(model_name)
    if model_name is None:
        raise KeyError('No such model')
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if with_classification_head:
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
    else:
        model = AutoModel.from_pretrained(model_name)
    model = model.to(device)
    return model, tokenizer


def extract_features(dataset, model, tokenizer, tokens_fp, embeddings_fp, bs=16):
    embeddings = []
    for batch in tqdm(dataset.iter(bs), total=len(dataset) // bs):
        with torch.no_grad():
            rv = model(**{k: v for k, v in batch.items() if k in ['input_ids', 'token_type_ids', 'attention_mask']})

        with jsonlines.open(tokens_fp, mode='a') as writer:
            for i in range(batch['attention_mask'].shape[0]):
                mask = batch['attention_mask'][i] == 1
                writer.write({
                    'tokens': tokenizer.convert_ids_to_tokens(batch['input_ids'][i][mask]),
                    'offset_mapping': batch['offset_mapping'][i][mask].tolist(),
                })
                embeddings.append(rv.last_hidden_state[i][mask].cpu().numpy())
    np.save(embeddings_fp, np.asarray(embeddings))


def train(dataset, model, tokenizer, models_path, epochs=5, wr=0.3, bs=16, wd=0.1, save_ratio=4):
    eval_steps = len(dataset['train']) // bs // save_ratio
    training_args = TrainingArguments(
        output_dir=os.path.join(models_path, f'no_pretrain_4_context_wr{wr}_e{epochs}_bs{bs}_wd{wd}'),
        evaluation_strategy="steps",
        eval_steps=eval_steps,
        num_train_epochs=epochs,
        warmup_ratio=wr,
        per_device_train_batch_size=bs,
        weight_decay=wd,
        save_steps=eval_steps,
        save_strategy="steps", )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['test'],
        compute_metrics=compute_metrics,
        tokenizer=tokenizer
    )
    trainer.train()
