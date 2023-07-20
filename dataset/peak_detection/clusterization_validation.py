import os
import re
import json
import argparse
from tqdm import tqdm
import numpy as np
from collections import defaultdict
from sklearn.metrics import precision_recall_fscore_support

from utils import clean_title, detect_laughs_in_subtitle, get_search_windows
from const import PEAK_DETECTION_VIDEOS
from plot_clustering import plot_experiment

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_root", type=str, default='../standup_dataset', help="Path to the dataset folder")
parser.add_argument("--n_top", type=int, default=2,
                    help='How many best clustering algorithms to consider')


def main(args):
    log_labeling = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    videos = [clean_title(v) for vs in PEAK_DETECTION_VIDEOS.values() for v in vs]
    meta_data = json.load(open(os.path.join(args.dataset_root, 'meta_data.json')))

    spans_folder = os.path.join(args.dataset_root, 'experiments', 'detected_peaks')
    experiments_folder = os.path.join(args.dataset_root, 'experiments', 'clusterization_experiments')
    logs_folder = os.path.join(experiments_folder, 'clusterization_logs')
    plots_folder = os.path.join(experiments_folder, 'clusterization_plots')
    clustering_labels_folder = os.path.join(experiments_folder, 'clusterization_labels')
    annotated_labeled_subtitles_folder = os.path.join(args.dataset_root, 'preprocessed_sub', 'subtitles_faligned_annotation_labeled')
    for video_name in videos:
        fn = video_name + '.json'
        annotated_subtitles = json.load(open(os.path.join(annotated_labeled_subtitles_folder, fn)))
        not_labeled = [s[:3] for s in annotated_subtitles]
        search_windows = list(get_search_windows(not_labeled, max_duration=meta_data[video_name]['duration']))
        true = [s[3] for s in annotated_subtitles]
        reversed_fp = os.path.join(experiments_folder, 'reverse_labels', video_name + '.json')
        os.makedirs(os.path.join(experiments_folder, 'reverse_labels'), exist_ok=True)
        if os.path.isfile(reversed_fp):
            reverse_dict = json.load(open(reversed_fp))
        else:
            reverse_dict = defaultdict(dict)

        for experiment in tqdm(sorted(os.listdir(logs_folder))):
            logs = json.load(open(os.path.join(logs_folder, experiment, video_name + '.json')))

            top_alg = []
            reversed = []
            if experiment in reverse_dict:
                top_alg = list(reverse_dict[experiment].keys())
                reversed = [alg for alg in top_alg if reverse_dict[experiment][alg]]
            else:
                for alg, values in sorted(logs.items(), key=lambda x: sum(x[1]) / len(x[1]), reverse=True)[:args.n_top]:
                    print(alg, '{:.3f}'.format(sum(values) / len(values)))
                    top_alg.append(alg)
                    to_reverse = int(input('Reverse?'))
                    reverse_dict[experiment][alg] = to_reverse
                    if to_reverse:
                        reversed.append(alg)

            labels_fp = os.path.join(clustering_labels_folder, experiment, video_name + '.json')
            labels = json.load(open(labels_fp))
            labels = {k: np.asarray(v) for k, v in labels.items()}
            for alg in top_alg:
                alg_labels = labels[f'{alg}_labels']
                if alg in reversed:
                    labels[f'{alg}_labels'] = 1 - alg_labels

            plot_fp = os.path.join(plots_folder, f'{video_name}_{experiment}_top2.png')
            plot_experiment(top_alg, labels, plot_fp)

            audio_regions = json.load(
                open(os.path.join(spans_folder, re.sub(r'_trim.*?(?=_|$)', r'', experiment), video_name + '.json')))
            for alg in top_alg:
                alg_labels = labels[f'{alg}_labels']
                filtered_peaks = [
                    peak
                    for peak, l in zip(audio_regions, alg_labels)
                    if l == 1
                ]
                pred = [
                    1 if len(list(detect_laughs_in_subtitle(filtered_peaks, cut_start, cut_end))) > 0 else 0
                    for (cut_start, cut_end) in search_windows
                ]
                precision, recall, f1, _ = precision_recall_fscore_support(true, pred, average='binary')
                log_labeling[experiment][alg][video_name]['precision'] = precision
                log_labeling[experiment][alg][video_name]['recall'] = recall
                log_labeling[experiment][alg][video_name]['f1'] = f1
        json.dump(reverse_dict, open(os.path.join(experiments_folder, 'reverse_labels', video_name + '.json'), 'w'))
    json.dump(log_labeling, open(os.path.join(experiments_folder, 'peak_detection_cluster_labeling_accuracy_logs.json'), 'w'))


if __name__ == '__main__':
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
