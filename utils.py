import os
import subprocess
import re
import pandas as pd
import json


def clean_title(title):
    title = re.sub(r'[|.,"/]', r'', title)
    title = re.sub(r'й', r'й', title)
    title = re.sub(r'ё', r'ё', title)
    title = re.sub(r'é', r'é', title)
    return title


def get_tier_by_name(textgrid_obj, tier_name):
    for tier in textgrid_obj:
        if tier.name == tier_name:
            return tier
    return None


def load_validation_data(standup_root):
    # TODO: replace
    with open(os.path.join(standup_root, 'annotation', 'titles.txt'), encoding='utf-8') as f:
        titles = f.read().splitlines()
    titles = [clean_title(t) for t in titles]

    dfs = [
        pd.read_csv(os.path.join(standup_root, 'annotation', f'standup{i}.txt'),
                    sep='\t',
                    names=['tier', 'start', 'end', 'duration'],
                    usecols=[0, 2, 3, 4], dtype={'tier': str, 'start': float, 'end': float, 'duration': float})
        for i in range(len(titles))
    ]
    for i, df in enumerate(dfs):
        df['video_name'] = titles[i]

    annotations = pd.concat(dfs, axis=0)
    return titles, annotations


def get_documents(root):
    subtitles_annotated_folder = os.path.join(root, 'subtitles_faligned_labeled')
    documents = {}
    for fn in os.listdir(subtitles_annotated_folder):
        video_name = os.path.splitext(fn)[0]
        if video_name in documents:
            continue
        subtitles = json.load(open(os.path.join(subtitles_annotated_folder, fn)))
        documents[video_name] = subtitles
    return documents


def get_splits_audio_spans_labels(documents, video_names, window=5):
    split_spans_and_labels = []
    for d in sorted(video_names):
        subs = documents[d]
        split_spans_and_labels.extend([
            [d, j, [s['audio_span'] for s in split], split[-1]['label']]
            for j, split in enumerate(zip(*[subs[i:] for i in range(window)]))
        ])
    split_labels = [s[-1] for s in split_spans_and_labels]
    return split_spans_and_labels, split_labels


def get_laughs_from_annotation(annotations, video_name, include_applause=False):
    if include_applause:
        mask = annotations.video_name == video_name
    else:
        mask = (annotations.video_name == video_name) & (annotations.tier == 'laughter')
    true_laughs = annotations[mask]
    laughs = [(r['start'], r['end']) for r in true_laughs.to_dict('records')]
    return laughs


def interval_overlap(interval0, interval1):
    overlap = max([
        0,
        min([interval0[1], interval1[1]]) - max([interval0[0], interval1[0]])
    ])
    return overlap


def cut_segment(fp, start, end, temp_fp='', rm=True, ext='mp4', with_codec=False):
    duration = end - start
    temp_fp = os.path.join(temp_fp, f'temp.{ext}')
    if with_codec:
        rv = subprocess.run(
            f'ffmpeg -ss {start} -i "{fp}" -to {duration} -c:a aac "{temp_fp}"  -hide_banner -loglevel error',
            shell=True, check=True, text=True)
    else:
        rv = subprocess.run(
            f'ffmpeg -ss {start} -i "{fp}" -to {duration} -c copy "{temp_fp}"  -hide_banner -loglevel error',
            shell=True, check=True, text=True)
    if rm:
        os.remove(temp_fp)


def norm_subtitles_spans(sentences):
    sentences = [[round(s[0], 2), round(s[1], 2), s[2]] for s in sentences]
    for s0, s1 in zip(sentences, sentences[1:]):
        if s0[1] > s1[0]:
            start, end = s1[0], s0[1]
            s0[1] = start
            s1[0] = end
    return sentences


def get_search_windows(subtitles, max_duration, max_window_length=0.7, min_pause_length=0.2):
    for ((_, end, _), (start, _, _)) in zip(subtitles, subtitles[1:]):
        duration = start - end
        if duration > min_pause_length:
            yield (end, start)
        else:
            yield (end, min([end + max_window_length, max_duration]))
    end = subtitles[-1][1]
    yield (end, min([end + max_window_length, max_duration]))


def detect_laughs_in_subtitle(laughs, start, end):
    for s, e, t in laughs:
        if interval_overlap((s, e), (start, end)) > 0:
            yield (s, e)
