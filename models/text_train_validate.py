import os
import json
import argparse
import numpy as np
from tqdm import tqdm
import torch
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedShuffleSplit
from razdel import tokenize as rus_tokenize
from nltk.tokenize import word_tokenize as eng_tokenize

from bert import train as bert_train, load_model as load_bert_model, create_dataset as create_bert_dataset
from colbert import train as colbert_train, load_model as load_colbert_model, create_dataset as create_colbert_dataset
from utils import get_documents

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_root", type=str, default='../standup_dataset', help="Path to the dataset folder")
parser.add_argument("--models_path", type=str, default='models',
                    help="Path to the folder where models and their logs will be saved")
parser.add_argument("--model_name", type=str, default='svm',
                    help="One of the model names [svm, bert, rubert, colbert, bert_multilingual]")
parser.add_argument("--unfreeze_bert", action="store_true",
                    help="Unfreeze BERT in ColBERT training, default is freezed")
parser.add_argument("--cv_n_splits", type=int, default=4,
                    help='Number of splits in StratifiedShuffleSplit cross-validation')
parser.add_argument("--epochs", type=int, default=5, help='Number of epochs to train')
parser.add_argument("--wr", type=float, default=0.3, help='Warming up rate')
parser.add_argument("--bs", type=int, default=16, help='Batch size')
parser.add_argument("--wd", type=float, default=0.1, help='Weight decay')
parser.add_argument("--lr", type=float, default=5e-5, help='Learning rate')
parser.add_argument("--save_ratio", type=int, default=4, help='How many times save the results during an epoch')
parser.add_argument("--lang", type=str, default='RUS', help="Language for TFIDF tokenization")


def create_dataset(root, labeled=True):
    documents = get_documents(root)
    split_dataset = []
    window = 5
    for d, subs in sorted(documents.items()):
        split_dataset.extend([
            [[s['text'] for s in split], 0 if not labeled else split[-1]['label']]
            for split in zip(*[subs[i:] for i in range(window)])
        ])
    dataset_neg = [l for _, l in split_dataset if l == 0]
    dataset_pos = [l for _, l in split_dataset if l == 1]
    print('Number of examples:', len(split_dataset))
    print(f'Number of negative examples: {len(dataset_neg)} ({len(dataset_neg) / len(split_dataset):.2f})')
    print(f'Number of positive examples: {len(dataset_pos)} ({len(dataset_pos) / len(split_dataset):.2f})')
    return split_dataset


def run_svm(split_dataset, n_splits, tokenizer_func, models_path):
    svm_dataset = [[' '.join(' '.join(tokenizer_func(s)) for s in subs), l] for subs, l in
                   split_dataset]
    svm_texts, svm_labels = list(zip(*svm_dataset))
    sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=0.2, random_state=0)
    logs_pred = []
    logs_chance = []
    logs_ones = []
    X = np.asarray(svm_texts)
    y = np.asarray(svm_labels)
    for train_indx, test_indx in tqdm(sss.split(svm_texts, svm_labels), total=n_splits):
        X_train, X_test, y_train, y_test = X[train_indx], X[test_indx], y[train_indx], y[test_indx]
        transformer = TfidfVectorizer()
        X_train = transformer.fit_transform(X_train)
        X_test = transformer.transform(X_test)
        clf = svm.SVC()
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        logs_pred.append(list(precision_recall_fscore_support(y_test, y_pred, average='binary'))[:3] + [
            accuracy_score(y_test, y_pred)])
        logs_ones.append(list(precision_recall_fscore_support(y_test, np.ones(y_test.shape), average='binary'))[:3] + [
            accuracy_score(y_test, np.ones(y_test.shape))])
        rand_pred = np.random.randint(2, size=y_test.shape[0])
        logs_chance.append(list(precision_recall_fscore_support(y_test, rand_pred, average='binary'))[:3] + [
            accuracy_score(y_test, rand_pred)])
    json.dump(logs_pred, open(os.path.join(models_path, 'svm_logs.json'), 'w'))
    json.dump(logs_ones, open(os.path.join(models_path, 'ones_logs.json'), 'w'))
    json.dump(logs_chance, open(os.path.join(models_path, 'chance_logs.json'), 'w'))


def main(args):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dataset = create_dataset(args.dataset_root)
    models_path = os.path.join(args.dataset_root, args.models_path, args.model_name)
    os.makedirs(models_path, exist_ok=True)
    if 'svm' == args.model_name:
        if args.lang == 'RUS':
            tokenizer_func = lambda s: [_.text for _ in rus_tokenize(s)]
        else:
            tokenizer_func = lambda s: eng_tokenize(s)
        run_svm(dataset, args.cv_n_splits, tokenizer_func, models_path)
        return
    elif 'colbert' == args.model_name:
        model, tokenizer = load_colbert_model(device, freeze_bert=not args.unfreeze_bert)
        dataset = create_colbert_dataset(dataset, tokenizer, device)
    elif 'bert' in args.model_name:
        model, tokenizer = load_bert_model(args.model_name, device)
        dataset = create_bert_dataset(dataset, tokenizer, device)
    else:
        raise ValueError('No such model')

    if 'colbert' == args.model_name:
        colbert_train(dataset, model, device, models_path, epochs=args.epoch, wr=args.wr, bs=args.bs, wd=args.wd,
                      lr=args.lr, save_ratio=args.save_ratio)
    elif 'bert' in args.model_name:
        bert_train(dataset, model, tokenizer, models_path, epochs=args.epochs, wr=args.wr, bs=args.bs,
                   wd=args.wd,
                   save_ratio=args.save_ratio)


if __name__ == '__main__':
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
