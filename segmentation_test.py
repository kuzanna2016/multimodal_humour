import os
import json
import difflib
import numpy as np

from utils import clean_title
from resegment_subtitles import main


def compare_sentences(sent0, sent1, return_values=False):
    d = difflib.Differ()
    if len(sent1) == 0:
        return 0, 0, 0, 0
    diff = list(d.compare(sent0, sent1))
    pos = [d[2:] for d in diff if d.startswith('+')]
    neg = [d[2:] for d in diff if d.startswith('-')]
    unk = [d[2:] for d in diff if d.startswith('?')]
    print('Error', len(pos) / len(sent1))
    if return_values:
        return pos, neg, unk
    i_counter = 0
    for d in diff:
        if (d.startswith('+')):
            i_counter += 1
        if not (d.startswith(' ')):
            print(d)
        if i_counter >= len(pos) - 1:
            break
    print('=' * 30)
    return len(pos), len(neg), len(unk), len(sent1)


if __name__ == '__main__':
    standup_root = '..'
    aligned_folder = os.path.join(standup_root, 'mfa_data/standup_rus_aligned_beam100_retry_beam400')
    sub_folder = os.path.join(standup_root, 'sub')
    metadata = json.load(open('../meta_data.json', encoding='utf-8'))
    metadata = {clean_title(k): v for k, v in metadata.items()}
    black_list = json.load(open('../black_list.json', encoding='utf-8'))
    gold_merged_folder = '../subtitles_merged_splitted_gold'
    with open('../annotation/titles.txt', encoding='utf-8') as f:
        titles = f.read().splitlines()
    titles = [clean_title(t) for t in titles]

    videos_sorted = sorted(list(metadata.keys()))
    videos = []
    for video_name in os.listdir(gold_merged_folder):
        video_name = os.path.splitext(video_name)[0]
        video_name = clean_title(video_name)
        if video_name in black_list or video_name not in metadata:
            continue
        videos.append(video_name)
    videos.extend(titles)
    videos = sorted(list(set(videos)))
    aligned_videos = main(videos, sub_folder, aligned_folder, videos_sorted)

    stats = {}
    for video_name in videos:
        print(video_name)
        print('=' * 30)
        aligned_video = aligned_videos[video_name]
        merged_sentences = [s['text'] for s in aligned_video]
        fp = os.path.join(gold_merged_folder, video_name + '.json')
        if not os.path.isfile(fp):
            json.dump([[0, 0, merged_sentences]], open(fp, 'w'), ensure_ascii=False)
        gold_sentences = json.load(open(fp))
        gold_sentences = [s for b, e, sentences in gold_sentences for s in sentences]
        pos, neg, unk, tr = compare_sentences(merged_sentences, gold_sentences[:200])
        stats[video_name] = {'pos': pos, 'neg': neg, 'unk': unk, 'tr': tr}
    print(len(stats))
    error_rate = []
    for video_name, s in stats.items():
        print(video_name)
        print('{:.2f}'.format(1 - s['pos'] / s['tr']))
        error_rate.append(s['pos'] / s['tr'])
    print(f'Mean error rate is {1 - np.mean(error_rate):.2f}')
