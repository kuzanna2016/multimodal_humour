import os
import re
import scipy.signal as signal
import numpy as np
import pandas as pd
import soundfile as sf
import librosa
# import auditok
from tqdm.notebook import tqdm
import json
from collections import defaultdict
import random
import cv2
import jsonlines
import itertools

from sklearn.metrics import precision_recall_fscore_support, accuracy_score

import IPython
import IPython.display as ipd
import matplotlib.pyplot as plt
from math import ceil
import sklearn
from sklearn import svm
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from text_train_validate import get_documents
from sklearn.model_selection import StratifiedShuffleSplit
# from razdel import tokenize
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_roots", nargs="+", type=str, default=('../standup_dataset',),
                    help="Path to the dataset folders, one if only one dataset, more if multilingual dataset")
parser.add_argument("--langs", type=str, nargs="+", default=('RUS',), help="Languages for multilingual setting")
parser.add_argument("--model_root", type=str, default='models', help="Path to save the logs")
parser.add_argument("--n_splits", type=int, default=4,
                    help="Number of splits in StratifiedShuffleSplit cross-validation")


def get_splits_audio_spans_labels(documents, video_names, window=5):
    split_spans_and_labels = []
    for d in sorted(video_names):
        subs = documents[d]
        split_spans_and_labels.extend([
            [d, j, [s['audio_span'] for s in split], split[-1]['label']]
            for j, split in enumerate(zip(*[subs[i:] for i in range(window)]))
        ])
    split_labels = [s[-1] for s in split_spans_and_labels]
    return split_spans_and_labels, split_labels


def load_text_features(standup_root):
    embeddings_fp = os.path.join(standup_root, 'dataset', 'text_features', 'embeddings.npy')
    bert_embeddings = np.load(embeddings_fp, allow_pickle=True)
    bert_cls_embeddings = np.vstack([a[0] for a in bert_embeddings])
    return bert_cls_embeddings


def load_of_mean_features(standup_root):
    open_face_features = np.load(
        os.path.join(standup_root, 'dataset', 'facial_features', 'mean_context_utterance_features.npy'))
    return open_face_features


def load_mean_video_features(standup_root):
    video_features = np.load(os.path.join(standup_root, 'dataset', 'video_mean_features.npy'))
    return video_features


def load_lang_features(root):
    meta_data = json.load(open(os.path.join(root, 'meta_data.json')))
    documents = get_documents(root)
    _, split_labels = get_splits_audio_spans_labels(documents, sorted(meta_data.keys()))

    text_features = load_text_features(root)
    open_face_features = load_of_mean_features(root)
    video_features = load_mean_video_features(root)
    return text_features, open_face_features, video_features, split_labels


def cross_validation_round(config, X, train_indx, test_indx, y_train, y_test, logs_pred, multilingual=False,
                           eng_labels_test=[], rus_labels_test=[]):
    X_train, X_test = X[train_indx], X[test_indx]
    print('Number of features:', X_train.shape[1])
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    clf = svm.SVC()
    clf.fit(X_train_scaled, y_train)
    y_pred = clf.predict(X_test_scaled)
    print(classification_report(y_test, y_pred, digits=3))
    logs_pred['_'.join(config)].append(
        list(precision_recall_fscore_support(y_test, y_pred, average='binary'))[:3] + [
            accuracy_score(y_test, y_pred)])
    if multilingual:
        logs_pred['_'.join(config) + '_eng'].append(list(
            precision_recall_fscore_support(y_test[eng_labels_test], y_pred[eng_labels_test],
                                            average='binary'))[:3] + [
                                                        accuracy_score(y_test[eng_labels_test],
                                                                       y_pred[eng_labels_test])])
        logs_pred['_'.join(config) + '_rus'].append(list(
            precision_recall_fscore_support(y_test[rus_labels_test], y_pred[rus_labels_test],
                                            average='binary'))[:3] + [
                                                        accuracy_score(y_test[rus_labels_test],
                                                                       y_pred[rus_labels_test])])
    return logs_pred


def main(args):
    sss = StratifiedShuffleSplit(n_splits=args.n_splits, test_size=0.2, random_state=0)
    logs_pred = defaultdict(list)
    lang_features = []
    for dataset_root in args.dataset_roots:
        features = load_lang_features(dataset_root)
        lang_features.append(features)
    lang_separator = lang_features[0][0].shape[0]
    bert_cls_embeddings, open_face_features, video_features, split_labels = map(np.concatenate, zip(*lang_features))

    y = split_labels
    n_samples = y.shape[0]
    modalities = ['text', 'video', 'facial']
    X = np.empty(n_samples, dtype=float)
    models_folder = os.path.join(args.model_root, f'models_multimodal_{"|".join(args.langs)}_baseline')
    os.makedirs(models_folder, exist_ok=True)
    multilingual = len(args.langs) > 1
    for train_indx, test_indx in tqdm(sss.split(X, y), total=args.n_splits):
        y_train, y_test = y[train_indx], y[test_indx]
        eng_labels_test = test_indx < lang_separator if multilingual else []
        rus_labels_test = test_indx >= lang_separator if multilingual else []
        for r in range(1, len(modalities) + 1):
            for config in itertools.combinations(modalities, r):
                print('_'.join(config))
                X = np.empty((n_samples, 0), dtype=float)
                if 'text' in config:
                    X = np.hstack([X, bert_cls_embeddings])
                if 'video' in config:
                    X = np.hstack([X, video_features])
                if 'facial' in config:
                    X = np.hstack([X, open_face_features])
                logs_pred = cross_validation_round(config, X, train_indx, test_indx, y_train, y_test, logs_pred,
                                                   multilingual=multilingual,
                                                   eng_labels_test=eng_labels_test, rus_labels_test=rus_labels_test)
                json.dump(logs_pred, open(os.path.join(models_folder, 'cross_validation_logs.json'), 'w'))
    json.dump(logs_pred, open(os.path.join(models_folder, 'cross_validation_logs.json'), 'w'))


if __name__ == '__main__':
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
