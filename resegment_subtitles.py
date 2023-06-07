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

punctuation += '»«…—'
punctuation = punctuation.replace(',', '')

TOKENS_SPLIT = r'\s+|\b[+=,!?:;"»«…/.—-]+\b'

PREPROCESSING_SUBSTITUTIONS = [
    [r'(?<=\d)%(?=\s|$)', ' процентов', []],
    [r'(?<=\d)%-', ' процент', []],
    [r'(?<=\d)\s?ч(?=\s|$)', r' часов', []],
    [r'(?<=\d)(час|лет|кг)', r' {}', [1]],
]


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
            indices = np.concatenate([indices[:start], new_indices, indices[end:]]).flatten()
        assert len(indices) == len(text_sub)
        text = text_sub
    return text, indices

def garbage_clean(text):
    text = re.sub(r'\d+\s\d+:\d+:\d+,?\d*\s-->\s\d+:\d+:\d+,?\d*\s', r'', text, flags=re.IGNORECASE)
    text = re.sub(r'\d+:\d+:\d+:\d+\s-\s\d+:\d+:\d+:\d+\s(speaker\s\d+)?', r'', text, flags=re.IGNORECASE)
    text = re.sub(r'\s*\[\s*__\s*\]\s*', r' ', text)
    text = re.sub(r'’', r"'", text)
    text = re.sub(r'\s+', r' ', text.strip())
    text = re.sub(r'о-о-о', r'о о о', text, flags=re.IGNORECASE)
    text = re.sub(r'(\w)-(\1-?){2,}', r'\1', text, flags=re.IGNORECASE)
    text = re.sub(r'(?<=\w)\s?([.,!?…]+)\s?(?=\w|$)', r'\1 ', text)
    text = re.sub(r'(?<=\w)\s…', r'…', text)
    text = re.sub(ordinal_endings, r'\1\2', text)
    text = re.sub(r'(?<=\w):(["«])\s?(?=\w)', r': \1', text)
    text = text.strip()
    return text


def clean_text_with_mapping(text, for_mfa=True):
    indices, tokens = map_token_indices(text)
    text, indices = perform_substitutions(text, PREPROCESSING_SUBSTITUTIONS, indices)

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
        return tokens, tokens_map, indices
    return new_tokens, [], []


def add_new_phrase(phrases, current_tokens):
    for token in current_tokens:
        token.pop('isupper')
    phrases.append({
        'text': ' '.join([t['text'] for t in current_tokens]),
        'audio_span': (current_tokens[0]['audio_span'][0], current_tokens[-1]['audio_span'][1]),
        'token_spans': current_tokens
    })
    return [], '', phrases


def iterate_until_valid_token(i, tokens, forward=True):
    if forward:
        while i + 1 < len(tokens) and (not tokens[i + 1] or tokens[i + 1]['audio_span'] == (-1, -1)):
            i += 1
        search_token = tokens[i + 1]
    else:
        while i > 1 and (not tokens[i - 1] or tokens[i - 1]['audio_span'] == (-1, -1)):
            i -= 1
        search_token = tokens[i - 1]
    return search_token


def split_tokens_into_phrases(tokens, delta_punctuation=0.3, delta_no_punctuation=0.6):
    phrases = []
    current_tokens = []
    current_phrase_text = ''
    for token in tokens:
        if not token:
            continue
        token['isupper'] = token.get('text', ' ')[0].isupper()

    for i, token in enumerate(tokens):
        token_text = token.get('text', '')
        if i == 0:
            current_phrase_text = token_text
            token['text_span'] = (0, len(token_text))
            current_tokens.append(token)
            continue
        previous_token = tokens[i - 1]
        if not current_tokens and token['audio_span'] != (-1, -1) and (
                not previous_token or previous_token['audio_span'] == (-1, -1)):
            current_phrase_text = token_text
            token['text_span'] = (0, len(token_text))
            current_tokens.append(token)
            continue
        else:
            previous_token = iterate_until_valid_token(i, tokens, forward=False)

        if not token:
            if previous_token['text'][-1] in punctuation:
                current_tokens, current_phrase_text, phrases = add_new_phrase(phrases, current_tokens)
            continue
        elif token['audio_span'] == (-1, -1):
            next_token = iterate_until_valid_token(i, tokens, forward=True)
            next_token['text'] = token_text + ' ' + next_token.get('text', '')
            continue

        time_distance = token['audio_span'][0] - previous_token['audio_span'][1]
        if previous_token['text'][-1] in punctuation:
            if token.get('isupper', False):
                current_tokens, current_phrase_text, phrases = add_new_phrase(phrases, current_tokens)
            elif time_distance >= delta_punctuation:
                current_tokens, current_phrase_text, phrases = add_new_phrase(phrases, current_tokens)
        elif time_distance >= delta_no_punctuation:
            current_tokens, current_phrase_text, phrases = add_new_phrase(phrases, current_tokens)

        current_phrase_text += ' ' + token_text
        current_phrase_text = current_phrase_text.strip()
        token_start = len(current_phrase_text) - len(token_text)
        token_end = len(current_phrase_text)
        token['text_span'] = (token_start, token_end)
        current_tokens.append(token)
    if current_tokens:
        current_tokens, current_phrase_text, phrases = add_new_phrase(phrases, current_tokens)
    return phrases


def align_tokens_in_subtitles(subtitles, tg_aligned):
    words_intervals = get_tier_by_name(tg_aligned, 'words')
    words = [i for i in words_intervals if i.mark]

    aligned_tokens = []
    for subtitle_index, (start, end, text) in enumerate(subtitles):
        if start == end:
            continue
        cleaned_text = garbage_clean(text)
        text_tokens, tokens_mapping, tokens_indices = clean_text_with_mapping(cleaned_text, for_mfa=False)
        interval_words = [w for w in words if w.minTime < end and w.maxTime > start]
        interval_strs = [w.mark.lower() for w in interval_words]
        if not interval_strs:
            continue

        words_taken = set(w for ws in tokens_mapping.values() for w in ws)
        assert len(words_taken) >= len(interval_words)
        for t, ws in tokens_mapping.items():
            ws = [w for w in ws if w < len(interval_words)]
            if not ws:
                audio_span = (-1, -1)
            else:
                interval_start = min(w.minTime for i, w in enumerate(interval_words) if i in ws)
                interval_end = max(w.maxTime for i, w in enumerate(interval_words) if i in ws)
                audio_span = (interval_start, interval_end)
            aligned_tokens.append(
                {
                    'text': text_tokens[t].strip(),
                    'audio_span': audio_span,
                }
            )
        aligned_tokens.append({})
    return aligned_tokens


def postprocess_phrases(phrases):
    for phrase in phrases:
        text = phrase['text']
        if re.search(r'(?<=\w) -(?=\w)', text) is None:
            continue
        indices = np.zeros(len(text), dtype=int)
        for i, t in enumerate(phrase['token_spans']):
            indices[max(t['text_span'][0] - 1,0):t['text_span'][1]] = i
        sub_text, sub_indices = perform_substitutions(text, [[r'(?<=\w) -(?=\w)', r'-', []]], indices)
        phrase['text'] = sub_text
        for i, t in enumerate(phrase['token_spans']):
            ixs = np.where(sub_indices == i)[0]
            start = ixs.min()
            start = start + 1 if start > 0 and len(ixs) > len(t['text']) else start
            end = ixs.max() + 1
            start, end = start.item(), end.item()
            t['text_span'] = (start, end)
            assert sub_text[start:end] == t['text']
    return phrases

def main(videos, sub_folder, aligned_folder, videos_sorted=None, save_to=None):
    aligned_videos = {}
    for video_index, video_name in enumerate(tqdm(videos)):
        video_index = video_index if videos_sorted is None else videos_sorted.index(video_name)
        subtitles = json.load(open(os.path.join(sub_folder, video_name + '.json')))
        subtitles = norm_subtitles_spans(subtitles)

        textgrid_fp = os.path.join(aligned_folder, f"{video_index}.TextGrid")
        tg_aligned = textgrid.TextGrid()
        tg_aligned.read(textgrid_fp)

        aligned_tokens = align_tokens_in_subtitles(subtitles, tg_aligned)
        segmented_phrases = split_tokens_into_phrases(aligned_tokens)
        aligned_videos[video_name] = postprocess_phrases(segmented_phrases)
    if save_to is not None:
        for video_name, phrases in aligned_videos.items():
            json.dump(phrases, open(os.path.join(save_to, video_name + '.json'), 'w'), ensure_ascii=False)
    return aligned_videos


if __name__ == '__main__':
    standup_root = '..'
    aligned_folder = os.path.join(standup_root, 'mfa_data/standup_rus_aligned_beam100_retry_beam400')
    segmented_folder = os.path.join(standup_root, 'subtitles_faligned')
    sub_folder = os.path.join(standup_root, 'sub')
    metadata = json.load(open('../meta_data.json', encoding='utf-8'))
    videos = sorted(list(metadata.keys()))
    aligned_videos = main(videos, sub_folder, aligned_folder, save_to=segmented_folder)
