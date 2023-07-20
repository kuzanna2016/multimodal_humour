import os
import argparse
import matplotlib.pyplot as plt
import librosa
import json
from utils import get_laughs_from_annotation, load_validation_data, get_search_windows, detect_laughs_in_subtitle

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_root", type=str, default='../standup_dataset', help="Path to the dataset folder")
parser.add_argument("--include_applause", action="store_true", help="Include applause as positive")


def main(args):
    segmented_folder = os.path.join(args.dataset_root, 'preprocessed_sub', 'subtitles_faligned')
    annotated_subs_folder = segmented_folder + '_annotation_labeled'
    os.makedirs(annotated_subs_folder, exist_ok=True)
    meta_data = json.load(open(os.path.join(args.dataset_root, 'meta_data.json')))
    titles, annotation = load_validation_data(args.dataset_root)

    for video_name in titles:
        subtitles = json.load(open(os.path.join(segmented_folder, video_name + '.json')))
        subtitles = [[*phrase['audio_span'], phrase['text']] for phrase in subtitles]

        laughs = get_laughs_from_annotation(annotation, video_name, args.include_applause)

        labeled_subtitles = []
        search_windows = list(get_search_windows(subtitles, max_duration=meta_data[video_name]['duration']))
        assert len(search_windows) == len(subtitles)
        for (start, end, t), (cut_start, cut_end) in zip(subtitles, search_windows):
            laughs_in_the_cut = list(detect_laughs_in_subtitle(laughs, cut_start, cut_end))
            if not laughs_in_the_cut:
                labeled_subtitles.append([start, end, t, 0])
            else:
                labeled_subtitles.append([start, end, t, 1])
        json.dump(labeled_subtitles, open(os.path.join(annotated_subs_folder, video_name + '.json'), 'w'),
                  ensure_ascii=False)


if __name__ == '__main__':
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
