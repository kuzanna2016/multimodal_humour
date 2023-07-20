import os
import numpy as np
from tqdm.notebook import tqdm
import json
from collections import defaultdict
import itertools

from sklearn import svm
from sklearn.metrics import classification_report, precision_recall_fscore_support, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
import argparse
from utils import get_splits_audio_spans_labels, get_documents

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_roots", nargs="+", type=str, default=('../standup_dataset',),
                    help="Path to the dataset folders, one if only one dataset, more if multilingual dataset")
parser.add_argument("--langs", type=str, nargs="+", default=('RUS',), help="Languages for multilingual setting")
parser.add_argument("--model_root", type=str, default='models', help="Path to save the logs")
parser.add_argument("--n_splits", type=int, default=4,
                    help="Number of splits in StratifiedShuffleSplit cross-validation")


def load_text_features(dataset_root):
    embeddings_fp = os.path.join(dataset_root, 'features', 'bert_features', 'embeddings.npy')
    bert_embeddings = np.load(embeddings_fp, allow_pickle=True)
    bert_cls_embeddings = np.vstack([a[0] for a in bert_embeddings])
    return bert_cls_embeddings


def load_of_mean_features(dataset_root):
    open_face_features = np.load(
        os.path.join(dataset_root, 'features', 'facial_mean_context_utterance_features.npy'))
    return open_face_features


def load_mean_video_features(dataset_root, n_samples, meta_data):
    fp = os.path.join(dataset_root, 'features', 'video_mean_features.npy')
    if os.path.isfile(fp):
        video_features = np.load(fp)
    else:
        video_features = np.empty((n_samples, 768), dtype=float)
        i = 0
        for video_name in tqdm(sorted(meta_data.keys())):
            vf = np.load(os.path.join(dataset_root, 'features', 'video_features', video_name + '.npy'))
            mean_vector = vf.mean(axis=1)
            n_samples = mean_vector.shape[0]
            video_features[i:i + n_samples] = mean_vector
            i += n_samples
        np.save(os.path.join(dataset_root, 'dataset', 'video_mean_features.npy'), video_features)
    return video_features


def load_lang_features(root):
    meta_data = json.load(open(os.path.join(root, 'meta_data.json')))
    documents = get_documents(root)
    _, split_labels = get_splits_audio_spans_labels(documents, sorted(meta_data.keys()))

    text_features = load_text_features(root)
    open_face_features = load_of_mean_features(root)
    video_features = load_mean_video_features(root, len(split_labels))
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
