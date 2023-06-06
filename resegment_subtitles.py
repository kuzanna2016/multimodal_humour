import re
import os
import json
from collections import defaultdict

from tqdm import tqdm

from string import punctuation
import numpy as np
import textgrid
from utils import clean_title, get_tier_by_name
from swear_words import preproc_swear_word
from numeric import ordinal_endings, change_numeric


def norm_subtitles_spans(sentences):
    sentences = [[round(s[0], 2), round(s[1], 2), s[2]] for s in sentences]
    for s0, s1 in zip(sentences, sentences[1:]):
        if s0[1] > s1[0]:
            start, end = s1[0], s0[1]
            s0[1] = start
            s1[0] = end
    return sentences


def map_token_indices(text):
    indexes = np.zeros(len(text), dtype=int)
    tokens = []
    pointer = 0
    i = None
    for i, m in enumerate(re.finditer(TOKENS_SPLIT, text, flags=re.IGNORECASE)):
        start, end = m.span(0)
        indexes[max(pointer - 1, 0):start] = i
        tokens.append(text[max(pointer - 1, 0):start])
        pointer = end
    if i is not None:
        indexes[pointer - 1:] = i + 1
        tokens.append(text[pointer - 1:])
    else:
        tokens.append(text)
    return indexes, tokens


def get_span_insert(indices, start, end, insert_len):
    token_indices = indices[start:end]
    token_indices = np.unique(token_indices).tolist()
    assert len(token_indices) == 1
    token_index = token_indices[0]

    new_token_indices = [token_index] * insert_len
    return new_token_indices


def perform_substitutions(text, substitutions, indices):
    for substitution in substitutions:
        pattern, repl, groups = substitution

        repl_regex = repl
        for i in groups:
            repl_regex = repl_regex.replace('{}', fr'\{i}')
        text_sub = re.sub(pattern, repl_regex, text)

        inserts = []
        for m in re.finditer(pattern, text):
            start = m.start()
            end = m.end()

            sub_repl = repl[:]
            for i in groups:
                sub_repl = sub_repl.format(m.group(i))

            new_token_indices = get_span_insert(indices, start, end, len(sub_repl))
            inserts.append([start, new_token_indices, end])
        for start, new_indices, end in sorted(inserts, reverse=True):
            indices = np.concatenate([indices[:start], new_token_indices, indices[end:]]).flatten()
        assert len(indices) == len(text_sub)
        text = text_sub
    return text, indices


TOKENS_SPLIT = r'\s+|\b[+=,!?:;"»«…/.—-]+\b'

substitutions = [
    [r'(?<=\d)%(?=\s|$)', ' процентов', []],
    [r'(?<=\d)%-', ' процент', []],
    [r'(?<=\d)\s?ч(?=\s|$)', r' часов', []],
    [r'(?<=\d)(час|лет|кг)', r' {}', [1]],
]

def garbage_clean(text):
    text = re.sub(r'\d+\s\d+:\d+:\d+,?\d*\s-->\s\d+:\d+:\d+,?\d*\s', r'', text, flags=re.IGNORECASE)
    text = re.sub(r'\d+:\d+:\d+:\d+\s-\s\d+:\d+:\d+:\d+\s(speaker\s\d+)?', r'', text, flags=re.IGNORECASE)
    text = re.sub(r'\[\s*__\s*\]', r'', text)
    text = re.sub(r'’', r"'", text)
    text = re.sub(r'\s+', r' ', text)
    text = re.sub(r'о-о-о', r'о о о', text, flags=re.IGNORECASE)
    text = re.sub(r'(\w)-(\1-?){2,}', r'\1', text, flags=re.IGNORECASE)
    text = re.sub(ordinal_endings, r'\1\2', text)
    text = text.strip()
    return text

def clean_text_with_mapping(text, for_mfa=True):
    text_rv = text[:]

    indices, tokens = map_token_indices(text)
    text, indices = perform_substitutions(text, substitutions, indices)

    indices_words, words = map_token_indices(text)
    assert len(indices) == len(indices_words)
    tokens_map = defaultdict(set)
    for t, w in zip(indices, indices_words):
        tokens_map[t].add(w)

    new_tokens = []
    backwards_step = 0
    for i, w in enumerate(words):
        i -= backwards_step
        w = re.sub(r'^[^\w*]+|[^*\w]+$', r'', w.strip())
        if '*' in w:
            w = preproc_swear_word(w)
            w = re.sub(r'^[^\w]+|[^\w]+$', r'', w.strip())
        w = re.sub(r'(\w)([уыаоэ])\2+$', r'\1\2', w.strip(), flags=re.IGNORECASE)
        w = re.sub(r'^([уыаоэяию])\1+$', r'\1', w, flags=re.IGNORECASE)
        w = change_numeric(w)
        n_new_splits = len(re.split(TOKENS_SPLIT, w, flags=re.IGNORECASE))
        if n_new_splits > 1:
            for t, ws in tokens_map.items():
                ws = [w + n_new_splits - 1 if w > i else w for w in ws]
                if i in ws:
                    ws.extend(list(range(i + 1, i + n_new_splits)))
                tokens_map[t] = ws
        if w:
            if w in punctuation:
                for t, ws in tokens_map.items():
                    if i in ws:
                        ws.remove(i)
                    ws = [w - 1 if w > i else w for w in ws]
                    tokens_map[t] = ws
                backwards_step += 1
                continue
            new_tokens.append(w)
        else:
            for t, ws in tokens_map.items():
                if i in ws:
                    ws.remove(i)
                ws = [w - 1 if w > i else w for w in ws]
                tokens_map[t] = ws
            backwards_step += 1

    if not for_mfa:
        return text_rv, tokens_map, indices
    return ' '.join(new_tokens), [], []


def main(videos, sub_folder, aligned_folder):
    aligned_videos = {}
    for video_index, video_name in enumerate(tqdm(videos)):
        subtitles = json.load(open(os.path.join(sub_folder, video_name + '.json')))
        subtitles = norm_subtitles_spans(subtitles)

        textgrid_fp = os.path.join(aligned_folder, f"{video_index}.TextGrid")
        tg_aligned = textgrid.TextGrid()
        tg_aligned.read(textgrid_fp)
        words_intervals = get_tier_by_name(tg_aligned, 'words')
        words = [i for i in words_intervals if i.mark]

        for subtitle_index, (start, end, text) in enumerate(subtitles):
            if start == end:
                continue
            text = garbage_clean(text)
            cleaned_text, tokens_mapping, tokens_indices = clean_text_with_mapping(text, for_mfa=False)
            interval_words = [w for w in words if w.minTime < end and w.maxTime > start]
            interval_strs = [w.mark.lower() for w in interval_words]
            if not interval_strs:
                continue

            words_taken = set(w for ws in tokens_mapping.values() for w in ws)
            assert len(words_taken) >= len(interval_words)
            tokens_mapping = {t: [w for w in ws if w < len(interval_words)] for t, ws in tokens_mapping.items()}




if __name__ == '__main__':
    standup_root = '..'
    aligned_folder = os.path.join(standup_root, 'mfa_data/standup_rus_aligned_beam100_retry_beam400')
    sub_splitted_folder = os.path.join(standup_root, 'subtitles_merged_splitted')
    sub_folder = os.path.join(standup_root, 'sub')
    metadata = json.load(open('../meta_data.json', encoding='utf-8'))
    metadata = {clean_title(k): v for k, v in metadata.items()}
    videos = sorted(metadata.keys())
    main(videos, sub_folder, aligned_folder)
