import os
import subprocess
import re
import pandas as pd


def clean_title(title):
    title = re.sub(r'[|.,"/]', r'', title)
    title = re.sub(r'й', r'й', title)
    title = re.sub(r'ё', r'ё', title)
    title = re.sub(r'é',r'é', title)
    return title


def get_tier_by_name(textgrid_obj, tier_name):
    for tier in textgrid_obj:
        if tier.name == tier_name:
            return tier
    return None


def load_validation_data(standup_root):
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


def cut_segment(fp, start, end, temp_fp='', play=True, rm=True, ext='mp4', with_codec=False):
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
