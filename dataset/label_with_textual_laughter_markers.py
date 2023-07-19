import os
import json
import argparse
from utils import get_search_windows, detect_laughs_in_subtitle

parser = argparse.ArgumentParser()

parser.add_argument("--dataset_root", type=str, default='../standup_dataset', help="Path to the dataset folder")


def main(args):
    metadata = json.load(open(os.path.join(args.dataset_root, 'meta_data.json'), encoding='utf-8'))
    subtitles_faligned_fp = os.path.join(args.dataset_root, 'subtitles_faligned_labeled')
    extracted_laughter_from_sub_fp = os.path.join(args.dataset_root, 'textual_laughter_markers')
    logs = []
    for video_name in metadata:
        subs = json.load(open(os.path.join(subtitles_faligned_fp, video_name + '.json')))
        laughs = json.load(open(os.path.join(extracted_laughter_from_sub_fp, video_name + '.json')))
        duration = metadata[video_name]['duration']

        subtitles = [[*phrase['audio_span'], phrase['text']] for phrase in subs]
        search_windows = list(get_search_windows(subtitles, max_duration=duration))
        for i, (_, (cut_start, cut_end)) in enumerate(zip(subtitles, search_windows)):
            laughs_in_the_cut = list(detect_laughs_in_subtitle(laughs, cut_start, cut_end))
            label = subs[i].get('label', 0)
            label_true = int(len(laughs_in_the_cut) > 0)
            logs.append((label, label_true))
            if laughs_in_the_cut:
                subs[i]['label'] = 1
        json.dump(subs, open(os.path.join(subtitles_faligned_fp, video_name + '.json'), 'w'), ensure_ascii=False)

    if __name__ == '__main__':
        args = parser.parse_args([] if "__file__" not in globals() else None)
        main(args)
