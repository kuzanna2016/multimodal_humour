import os
import json
from tqdm import tqdm
import argparse
from laughter_detection_model import set_up_ld_model, load, predict, cut_threshold
from utils import interval_overlap, cut_segment
from utils import get_search_windows

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_root", type=str, default='../standup_dataset', help="Path to the dataset folder")
parser.add_argument("--threshold", type=float, default=0.3, help='Laughter detection threshold')
parser.add_argument("--min_length", type=float, default=0.01, help='Minimum length of the detected laughter in seconds')

MIN_WINDOW_LENGTH_FOR_LD = 1.3


def main(args):
    audio_folder = os.path.join(args.dataset_root, 'vocal-remover')
    meta_data = json.load(open(os.path.join(args.dataset_root, 'meta_data.json')))

    temp_fp = 'temp.mp4'
    model, feature_fn, sample_rate, config, device = set_up_ld_model()
    output_path = os.path.join(args.dataset_root, 'subtitles_faligned_labeled')
    os.makedirs(output_path, exist_ok=True)

    for video_name in sorted(meta_data.keys()):
        audio_fp = os.path.join(audio_folder, video_name + '_Instruments.wav')
        subtitles_fp = os.path.join(args.dataset_root, 'preprocessed_sub', 'subtitles_faligned', video_name + '.json')
        subtitles = json.load(open(subtitles_fp))
        duration = meta_data[video_name].get('duration', subtitles[-1]['audio_span'][1])
        search_windows = list(
            get_search_windows([(*s['audio_span'], s['text']) for s in subtitles], max_duration=duration))
        labeled_subtitles = []
        for sub, (cut_start, cut_end) in tqdm(zip(subtitles, search_windows), total=len(subtitles)):
            cut_end_for_ld = max(cut_end, cut_start + MIN_WINDOW_LENGTH_FOR_LD)
            if cut_end_for_ld > duration:
                sub['label'] = 0
                labeled_subtitles.append(sub)
                continue

            cut_segment(audio_fp, cut_start, cut_end_for_ld, rm=False, with_codec=True)
            inference_generator = load(temp_fp, feature_fn, sample_rate, config)
            probs = predict(model, inference_generator, device)
            instances = cut_threshold(probs, args.threshold, args.min_length, temp_fp, log=0)
            instances = [(s, e) for s, e in instances if
                         interval_overlap((s + cut_start, e + cut_start), (cut_start, cut_end)) > 0]
            sub['label'] = 1 if instances else 0
            labeled_subtitles.append(sub)
            os.remove(temp_fp)
        json.dump(labeled_subtitles, open(os.path.join(output_path, video_name + '.json'), 'w'), ensure_ascii=False)


if __name__ == '__main__':
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
