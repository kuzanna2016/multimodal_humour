import os
import json
import argparse
from collections import defaultdict
from tqdm import tqdm
import subprocess
from laughter_detection_model import set_up_ld_model, load, predict, cut_threshold
from sklearn.metrics import precision_recall_fscore_support
from utils import interval_overlap, cut_segment, load_validation_data, get_search_windows

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_root", type=str, default='../standup_dataset', help="Path to the dataset folder")
parser.add_argument("--validation_type", type=str, default='window',
                    help='One of the [window, whole], to detect laughter either only in the window after each subtitle or the whole audio')


def window_cross_validation(model, pred, condition, search_windows, audio_fp, temp_fp, feature_fn, sample_rate,
                            config, device, duration, min_window_length_for_ld=1.3):
    for (cut_start, cut_end) in tqdm(search_windows):
        cut_end_for_ld = max(cut_end, cut_start + min_window_length_for_ld)
        if cut_end_for_ld > duration:
            for threshold in [0.1, 0.2, 0.3, 0.4, 0.5]:
                for min_length in [0.01, 0.05, 0.1, 0.15, 0.2]:
                    pred[f'th{threshold}_ml{min_length}_{condition}'].append(0)
            continue
        if condition == 'vocal-remover':
            cut_segment(audio_fp, cut_start, cut_end_for_ld, rm=False, with_codec=True)
        else:
            cut_segment(audio_fp, cut_start, cut_end_for_ld, rm=False)
        inference_generator = load(temp_fp, feature_fn, sample_rate, config)
        probs = predict(model, inference_generator, device)
        for threshold in [0.1, 0.2, 0.3, 0.4, 0.5]:
            for min_length in [0.01, 0.05, 0.1, 0.15, 0.2]:
                instances = cut_threshold(probs, threshold, min_length, temp_fp, log=0)
                instances = [(s, e) for s, e in instances if
                             interval_overlap((s + cut_start, e + cut_start), (cut_start, cut_end)) > 0]
                pred[f'th{threshold}_ml{min_length}_{condition}'].append(1 if instances else 0)
        os.remove(temp_fp)
    return pred


def whole_cross_validation(model, pred, condition, search_windows, audio_fp, temp_fp, feature_fn, sample_rate, config,
                           device):
    if condition == 'vocal-remover':
        subprocess.run(
            f'ffmpeg -i "{audio_fp}" -c:a aac "{temp_fp}"  -hide_banner -loglevel error',
            shell=True, check=True, text=True)
        audio_fp = temp_fp
    inference_generator = load(audio_fp, feature_fn, sample_rate, config)
    probs = predict(model, inference_generator, device)
    for threshold in [0.1, 0.2, 0.3, 0.4, 0.5]:
        for min_length in [0.01, 0.05, 0.1, 0.15, 0.2]:
            instances = cut_threshold(probs, threshold, min_length, audio_fp, log=0)
            pred_laughter = [
                any(interval_overlap((s, e), (cut_start, cut_end)) > 0 for s, e in instances)
                for (cut_start, cut_end) in search_windows
            ]
            pred[f'th{threshold}_ml{min_length}_{condition}'] = pred_laughter
    if condition == 'vocal-remover':
        os.remove(temp_fp)
    return pred


def main(args):
    audio_folder = os.path.join(args.dataset_root, 'audio')
    meta_data = json.load(open(os.path.join(args.dataset_root, 'meta_data.json')))
    titles, annotations = load_validation_data(args.dataset_root)
    validation_type = args.validation_type

    temp_fp = os.path.join(args.dataset_root, f'temp_{validation_type}.mp4')
    log = defaultdict(lambda: defaultdict(list))
    model, feature_fn, sample_rate, config, device = set_up_ld_model()
    for condition in ['original', 'vocal-remover']:
        for video_name in titles:
            if condition == 'original':
                audio_fp = os.path.join(audio_folder, video_name + '.mp4')
            else:
                audio_fp = os.path.join(args.dataset_root, 'vocal-remover', video_name + '_Instruments.wav')
            subtitles_fp = os.path.join(args.dataset_root, 'subtitles_faligned_annotation_labeled',
                                        video_name + '.json')
            annotated_subtitles = json.load(open(subtitles_fp))
            subtitles = [s[:3] for s in annotated_subtitles]

            pred = defaultdict(list)
            true = [s[3] for s in annotated_subtitles]

            duration = meta_data[video_name].get('duration', subtitles[-1][1])
            search_windows = list(get_search_windows(subtitles, max_duration=duration))
            if validation_type == 'window':
                pred = window_cross_validation(model, pred, condition, search_windows, audio_fp, temp_fp, feature_fn,
                                               sample_rate,
                                               config, device, duration)
            elif validation_type == 'whole':
                pred = whole_cross_validation(model, pred, condition, search_windows, audio_fp, temp_fp, feature_fn,
                                              sample_rate, config, device)
            else:
                raise ValueError(f'No validation type {validation_type}')
            for k, predictions in pred.items():
                precision, recall, f1, _ = precision_recall_fscore_support(true, predictions, average='binary')
                log[k]['precision'].append(precision)
                log[k]['recall'].append(recall)
                log[k]['f1'].append(f1)
    json.dump(log, open(
        os.path.join(args.dataset_root, f'laughter_detection_in_subtitle_{validation_type}_cross_validation.json'),
        'w'))


if __name__ == '__main__':
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
