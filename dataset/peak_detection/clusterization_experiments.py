import os
import json
import argparse
from tqdm import tqdm
import auditok
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering

from extract_audio_features import load_model, get_features
from utils import clean_title, get_laughs_from_annotation, load_validation_data, detect_laughs_in_subtitle
from const import PEAK_DETECTION_VIDEOS
from plot_clustering import plot_experiment

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_root", type=str, default='../standup_dataset', help="Path to the dataset folder")
parser.add_argument("--trim", type=float, default=0.5,
                    help='Maximum duration of detected peak to use for clusterization in seconds')


def cluster(last_hidden_states):
    labels = {}
    kmeans = KMeans(n_clusters=2, random_state=0, n_init="auto").fit(last_hidden_states)
    labels['kmeans_labels'] = kmeans.labels_

    clustering = AgglomerativeClustering(n_clusters=2, linkage='average')
    clustering_labels = clustering.fit_predict(last_hidden_states)
    labels['clustering_avg_labels'] = clustering_labels

    clustering = AgglomerativeClustering(n_clusters=2, linkage='ward')
    clustering_labels = clustering.fit_predict(last_hidden_states)
    labels['clustering_ward_labels'] = clustering_labels

    clustering = AgglomerativeClustering(n_clusters=2, linkage='complete')
    clustering_labels = clustering.fit_predict(last_hidden_states)
    labels['clustering_cmp_labels'] = clustering_labels

    clustering = AgglomerativeClustering(n_clusters=2, linkage='single')
    clustering_labels = clustering.fit_predict(last_hidden_states)
    labels['clustering_sng_labels'] = clustering_labels

    pca = PCA(n_components=2)
    reduced_states = pca.fit_transform(last_hidden_states)
    labels['pca_reduced_states'] = reduced_states
    labels['pca_explained_variance_ratio'] = pca.explained_variance_ratio_

    kmeans = KMeans(n_clusters=2, random_state=0, n_init="auto").fit(reduced_states)
    labels['kmeans_pca_labels'] = kmeans.labels_
    return labels, [k.replace('_labels', '') for k in labels if k.endswith('_labels')]

def label_regions(laughs, audio_regions, trim):
  labels = []
  for i, (start, end) in enumerate(audio_regions):
    if trim > 0:
      end = start + trim if end - start > trim else end
    if len(list(detect_laughs_in_subtitle(laughs, start, end))) > 0:
      labels.append(1)
    else:
      labels.append(0)
  return np.asarray(labels)


def main(args):
    spans_folder = os.path.join(args.dataset_root, 'detected_peaks')
    experiments_folder = os.path.join(args.dataset_root, 'clusterization_experiments')
    peaks_features_folder = os.path.join(experiments_folder, 'peaks_features')
    clustering_labels_folder = os.path.join(experiments_folder, 'clusterization_labels')
    plots_folder = os.path.join(experiments_folder, 'clusterization_plots')
    logs_folder = os.path.join(experiments_folder, 'clusterization_logs')
    for folder in [experiments_folder, peaks_features_folder, clustering_labels_folder, plots_folder, logs_folder]:
        os.makedirs(folder, exist_ok=True)

    videos = [clean_title(v) for vs in PEAK_DETECTION_VIDEOS.values() for v in vs]
    audio_folder = os.path.join(args.dataset_root, 'audio_subtracted')
    _, annotations = load_validation_data(args.dataset_root)

    processor, model, device = load_model()

    for video_name in videos:
        file_path_wav = os.path.join(audio_folder, video_name + '.wav')
        region = auditok.load(file_path_wav)
        laughs = get_laughs_from_annotation(args.dataset_root, video_name)
        for folder in tqdm(os.listdir(spans_folder)):
            experiment = folder + f'_trim{args.trim}'
            audio_regions = json.load(open(os.path.join(spans_folder, folder, video_name + '.json')))
            features_fp = os.path.join(peaks_features_folder, experiment, video_name + '.npy')
            if os.path.isfile(features_fp):
                last_hidden_states = np.load(features_fp)
            else:
                last_hidden_states = get_features(audio_regions, region, device, model, processor, trim=args.trim)
                os.makedirs(os.path.join(peaks_features_folder, experiment), exist_ok=True)
                np.save(features_fp, last_hidden_states)

            labels_fp = os.path.join(clustering_labels_folder, experiment, video_name + '.json')
            if os.path.isfile(labels_fp):
                labels = json.load(open(labels_fp))
                algs = [k.replace('_labels', '') for k in labels if k.endswith('_labels')]
                labels = {k: np.asarray(v) for k, v in labels.items()}
            else:
                labels, algs = cluster(last_hidden_states)
                true_labels = label_regions(laughs, audio_regions, args.trim)
                labels['true_labels'] = true_labels
                os.makedirs(os.path.join(clustering_labels_folder, experiment), exist_ok=True)
                json.dump({k: v if isinstance(v, list) else v.tolist() for k, v in labels.items()},
                          open(labels_fp, 'w'))

            plot_fp = os.path.join(plots_folder, f'{video_name}_{experiment}.png')
            if os.path.isfile(plot_fp):
                continue
            logs = plot_experiment(algs, labels, plot_fp)
            os.makedirs(os.path.join(logs_folder, experiment), exist_ok=True)
            json.dump(logs, open(os.path.join(logs_folder, experiment, video_name + '.json'), 'w'))


if __name__ == '__main__':
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
