import os
import json
from tqdm import tqdm
from laughter_detection_model import set_up_ld_model, load, predict, cut_threshold
from utils import interval_overlap, cut_segment
from laughter_detection_cross_validation import get_search_windows


def main(standup_root, audio_folder, meta_data, threshold, min_length, min_window_length_for_ld=1.3):
    temp_fp = os.path.join(standup_root, 'temp.mp4')
    model, feature_fn, sample_rate, config, device = set_up_ld_model()

    for video_name in sorted(meta_data.keys()):
        audio_fp = os.path.join(audio_folder, video_name + '_Instruments.wav')
        subtitles_fp = os.path.join(standup_root, 'subtitles_faligned', video_name + '.json')
        subtitles = json.load(open(subtitles_fp))
        duration = meta_data[video_name].get('duration', subtitles[-1]['audio_span'][1])
        search_windows = list(
            get_search_windows([(*s['audio_span'], s['text']) for s in subtitles], max_duration=duration))
        labeled_subtitles = []
        for sub, (cut_start, cut_end) in tqdm(zip(subtitles, search_windows), total=len(subtitles)):
            cut_end_for_ld = max(cut_end, cut_start + min_window_length_for_ld)
            if cut_end_for_ld > duration:
                sub['label'] = 0
                labeled_subtitles.append(sub)
                continue

            cut_segment(audio_fp, cut_start, cut_end_for_ld, temp_fp=standup_root, rm=False, with_codec=True)
            inference_generator = load(temp_fp, feature_fn, sample_rate, config)
            probs = predict(inference_generator, device)
            instances = cut_threshold(probs, threshold, min_length, temp_fp, log=0)
            instances = [(s, e) for s, e in instances if
                         interval_overlap((s + cut_start, e + cut_start), (cut_start, cut_end)) > 0]
            sub['label'] = 1 if instances else 0
            labeled_subtitles.append(sub)
            os.remove(temp_fp)
        json.dump(labeled_subtitles, open(subtitles_fp, 'w'), ensure_ascii=False)


if __name__ == '__main__':
    standup_root = '/data/disk1/share/akuznetsova/standup_rus'
    audio_folder = os.path.join(standup_root, 'voice-remover')
    meta_data = json.load(open(os.path.join(standup_root, 'meta_data.json')))
    main(standup_root, audio_folder, meta_data, threshold=0.3, min_length=0.01)