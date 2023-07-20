import os
import json
import argparse
from collections import defaultdict
from tqdm import tqdm
import auditok
from sklearn.metrics import precision_recall_fscore_support
from const import PEAK_DETECTION_VIDEOS
from utils import detect_laughs_in_subtitle, get_search_windows, clean_title

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_root", type=str, default='../standup_dataset', help="Path to the dataset folder")
parser.add_argument("--min_dur", type=float, default=0.3, help='Minimum duration of a valid audio event in seconds')
parser.add_argument("--max_dur", type=float, default=9, help='Maximum duration of an event in seconds')
parser.add_argument("--max_silence", type=float, default=0.5,
                    help='Maximum duration of tolerated continuous silence within an event in seconds')


def merge_regions(audio_regions, max_silence=0):
    region0 = audio_regions[0]
    merged_regions = [[region0.meta.start, region0.meta.end]]
    for r0, r1 in zip(audio_regions, audio_regions[1:]):
        if r1.meta.start - merged_regions[-1][1] <= max_silence:
            merged_regions[-1][1] = r1.meta.end
        else:
            merged_regions.append([r1.meta.start, r1.meta.end])
    return merged_regions


def main(args):
    logs = defaultdict(lambda: defaultdict(dict))
    spans_folder = os.path.join(args.dataset_root, 'detected_peaks')
    os.makedirs(spans_folder, exist_ok=True)
    meta_data = json.load(open(os.path.join(args.dataset_root, 'meta_data.json')))
    annotated_labeled_subtitles_folder = os.path.join(args.dataset_root, 'subtitles_faligned_annotation_labeled')
    audio_folder = os.path.join(args.dataset_root, 'audio_subtracted')

    videos = [clean_title(v) for vs in PEAK_DETECTION_VIDEOS.values() for v in vs]

    for video_name in videos:
        fn = video_name + '.json'
        annotated_subtitles = json.load(open(os.path.join(annotated_labeled_subtitles_folder, fn)))
        not_labeled = [s[:3] for s in annotated_subtitles]
        search_windows = list(get_search_windows(not_labeled, max_duration=meta_data[video_name]['duration']))
        true = [s[3] for s in annotated_subtitles]

        file_path = os.path.join(audio_folder, video_name + '.wav')
        for threshold in tqdm(range(10, 61), total=50):
            params = f'peaks_t{threshold}_mind{args.min_dur}_maxd{args.max_dur}_ms{args.max_silence}'
            audio_regions = list(auditok.split(
                file_path,
                min_dur=args.min_dur,
                max_dur=args.max_dur,
                max_silence=args.max_silence,
                energy_threshold=threshold
            ))
            if audio_regions:
                audio_regions = merge_regions(audio_regions, max_silence=args.max_silence)
            save_path = os.path.join(spans_folder, params)
            os.makedirs(save_path, exist_ok=True)
            json.dump(audio_regions, open(os.path.join(save_path, video_name + '.json'), 'w'))

            pred = [
                1 if len(list(detect_laughs_in_subtitle(audio_regions, cut_start, cut_end))) > 0 else 0
                for (cut_start, cut_end) in search_windows
            ]
            precision, recall, f1, _ = precision_recall_fscore_support(true, pred, average='binary')
            logs[params][video_name] = {
                'precision': precision,
                'recall': recall,
                'f1': f1
            }
    json.dump(logs, open(os.path.join(args.dataset_root, 'peak_detection_no_cluster_labeling_accuracy_logs.json'), 'w'))


if __name__ == '__main__':
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
