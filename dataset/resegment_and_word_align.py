import re
import os
import json
from collections import defaultdict
from tqdm import tqdm
from num2words import num2words

from string import punctuation
import numpy as np
import textgrid
from utils import get_tier_by_name, norm_subtitles_spans
from swear_words import preproc_swear_word
from numeric import ordinal_endings, change_numeric_rus, maybe_to_numeric

punctuation += '»«…—'
punctuation = punctuation.replace(',', '')

TOKENS_SPLIT = r'\s+|\b[+=,!?:;"»«…/.—-]+\b'

PREPROCESSING_SUBSTITUTIONS = [
    [r'(?<=\d)%(?=\s|$)', ' процентов', []],
    [r'(?<=\d)%-', ' процент', []],
    [r'(?<=\d)\s?ч(?=\s|$)', r' часов', []],
    [r'(?<=\d)(час|лет|кг)', r' {}', [1]],
]

NUMBER_REGEXP = r"\d+(?:[,\d]*\d)?(?:\.\d+)?"


def convert_number_to_word(n):
    if '911' in n:
        n = n.replace('911', 'nine one one')

    if '$' in n:
        n = n.replace('$', '')

        text = re.sub(NUMBER_REGEXP, lambda x: num2words(maybe_to_numeric(x.group(0)), lang='en') + ' dollars', n)
        return text

    if ':' in n:
        h, m = n.split(':')
        text = re.sub(NUMBER_REGEXP, lambda x: num2words(maybe_to_numeric(x.group(0)), lang='en'), h)
        text += ' ' + re.sub(NUMBER_REGEXP,
                             lambda x: num2words(maybe_to_numeric(x.group(0)), lang='en') if maybe_to_numeric(
                                 x.group(0)) != 0 else '',
                             m)
        return text

    if re.search(r"1\d{3}", n) is not None:
        if re.search(r"\d\d00", n) is not None:
            n = re.sub(r'00', r' hundred', n)
            text = re.sub(NUMBER_REGEXP, lambda x: ' ' + num2words(maybe_to_numeric(x.group(0)), lang='en'), n)
        else:
            text = re.sub(r'\d{2}', lambda x: ' ' + num2words(maybe_to_numeric(x.group(0)), lang='en'), n)
        text = text.replace("'s", 's')
        text = text.replace('tys', 'ties')
        return text

    if '%' in n:
        n = n.replace('%', ' percent')
        text = re.sub(NUMBER_REGEXP, lambda x: num2words(maybe_to_numeric(x.group(0)), lang='en'), n)
        return text

    if re.search(r"\d'?s", n) is not None:
        text = re.sub(NUMBER_REGEXP, lambda x: num2words(maybe_to_numeric(x.group(0)), lang='en'), n)
        text = text.replace("'s", 's')
        text = text.replace('tys', 'ties')
        return text

    if re.search(r'\d"', n) is not None:
        text = re.sub(NUMBER_REGEXP, lambda x: num2words(maybe_to_numeric(x.group(0)), lang='en'), n)
        text = text.replace('"', ' inches')
        return text

    if re.search(r'\d+(?:th|st|nd|rd)', n) is not None:
        n = re.sub(r'(\d+)(?:th|st|nd|rd)', r'\1', n)
        text = re.sub(NUMBER_REGEXP, lambda x: num2words(maybe_to_numeric(x.group(0)), lang='en', to='ordinal'), n)
        return text

    if re.search(r'\d+[-.,!?]', n) is not None:
        text = re.sub(NUMBER_REGEXP, lambda x: ' ' + num2words(maybe_to_numeric(x.group(0)), lang='en'), n)
        return text
    text = re.sub(NUMBER_REGEXP, lambda x: f" {num2words(maybe_to_numeric(x.group(0)), lang='en')} ", n)
    return text


def map_token_indices(text):
    indexes = np.zeros(len(text), dtype=int)
    tokens = []
    pointer = 0
    i = None
    for i, m in enumerate(re.finditer(TOKENS_SPLIT, text, flags=re.IGNORECASE)):
        start, end = m.span(0)
        t = text[max(pointer - 1, 0):start]
        indexes[max(pointer - 1, 0):start] = i
        tokens.append(t)
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
    text = re.sub(r'[^\w\']', r'', text)
    text = re.sub(r'\s+', r' ', text.strip())
    text = text.strip().lower()
    return text


def clean_text_with_mapping(text, for_mfa=True):
    indices, tokens = map_token_indices(text)
    tokens = [garbage_clean(t) for t in tokens]
    return tokens, [], indices
    #
    # indices_words, words = map_token_indices(text)
    # assert len(indices) == len(indices_words)
    # tokens_map = defaultdict(set)
    # for t, w in zip(indices, indices_words):
    #     tokens_map[t].add(w)
    #
    # new_tokens = []
    # backwards_step = 0
    # for i, w in enumerate(words):
    #     i -= backwards_step
    #     w = re.sub(r'^[^\w*]+|[^*\w]+$', r'', w.strip())
    #     if '*' in w:
    #         w = preproc_swear_word(w)
    #         w = re.sub(r'^[^\w]+|[^\w]+$', r'', w.strip())
    #     w = re.sub(r'(\w)([уыаоэ])\2+$', r'\1\2', w.strip(), flags=re.IGNORECASE)
    #     w = re.sub(r'^([уыаоэяию])\1+$', r'\1', w, flags=re.IGNORECASE)
    #     w = change_numeric(w)
    #     n_new_splits = len(re.split(TOKENS_SPLIT, w, flags=re.IGNORECASE))
    #     if n_new_splits > 1:
    #         for t, ws in tokens_map.items():
    #             ws = [w + n_new_splits - 1 if w > i else w for w in ws]
    #             if i in ws:
    #                 ws.extend(list(range(i + 1, i + n_new_splits)))
    #             tokens_map[t] = ws
    #     if w:
    #         if w in punctuation:
    #             for t, ws in tokens_map.items():
    #                 if i in ws:
    #                     ws.remove(i)
    #                 ws = [w - 1 if w > i else w for w in ws]
    #                 tokens_map[t] = ws
    #             backwards_step += 1
    #             continue
    #         new_tokens.append(w)
    #     else:
    #         for t, ws in tokens_map.items():
    #             if i in ws:
    #                 ws.remove(i)
    #             ws = [w - 1 if w > i else w for w in ws]
    #             tokens_map[t] = ws
    #         backwards_step += 1
    #
    # if not for_mfa:
    #     return tokens, tokens_map, indices
    # return new_tokens, [], []


def add_new_phrase(phrases, current_tokens):
    for token in current_tokens:
        token.pop('isupper')
    valid_audio_spans = [t['audio_span'] for t in current_tokens if t['audio_span'] != (-1,-1)]
    if not valid_audio_spans:
        return [], '', phrases
    phrases.append({
        'text': ' '.join([t['text'] for t in current_tokens]),
        'audio_span': (valid_audio_spans[0][0], valid_audio_spans[-1][1]),
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
        if i == len(tokens) - 1:
            current_phrase_text += ' ' + token_text
            current_phrase_text = current_phrase_text.strip()
            token_start = len(current_phrase_text) - len(token_text)
            token_end = len(current_phrase_text)
            token['text_span'] = (token_start, token_end)
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
        interval_words = [w for w in words if w.minTime < end and w.maxTime > start and w.mark != '<eps>']
        interval_strs = [w.mark.lower() for w in interval_words]
        if not interval_strs:
            continue

        indices, tokens = map_token_indices(text)
        tokens_clean = [garbage_clean(t) for t in tokens]
        if len(tokens) == len(interval_words):
            for interval, t in zip(interval_words, tokens):
                aligned_tokens.append({
                    'text': t,
                    'audio_span': (interval.minTime, interval.maxTime),
                })
            continue
        i_t = 0
        i_w = 0
        while i_t < len(tokens) and i_w < len(interval_words):
            if not tokens_clean[i_t]:
                aligned_tokens.append({
                    'text': tokens[i_t].strip(),
                    'audio_span': (-1,-1),
                })
                i_t += 1
                continue
            if tokens_clean[i_t] == interval_strs[i_w]:
                interval = interval_words[i_w]
                aligned_tokens.append(
                    {
                        'text': tokens[i_t].strip(),
                        'audio_span': (interval.minTime, interval.maxTime),
                    }
                )
                i_t += 1
                i_w += 1
                continue
            if "'" in tokens_clean[i_t]:
                if ''.join(interval_strs[i_w:i_w+2]) == tokens_clean[i_t]:
                    aligned_tokens.append({
                        'text': tokens[i_t].strip(),
                        'audio_span': (interval_words[i_w].minTime, interval_words[i_w+1].maxTime),
                    })
                    i_t += 1
                    i_w += 2
                    continue
            if i_t + 1 < len(tokens):
                next_matches = [
                    (j_t, [j for j in range(i_w+1, len(interval_words))
                    if tokens_clean[j_t] == interval_strs[j]])
                    for j_t in range(i_t+1, len(tokens))
                ]
                if not any(m for i, m in next_matches):
                    next_i_w = len(interval_words)
                    next_i_t = len(tokens)
                else:
                    next_i_t, next_match = [(i, m) for i, m in next_matches if m][0]
                    next_i_w = next_match[0]
                interval_start = min(w.minTime for w in interval_words[i_w:next_i_w])
                interval_end = max(w.maxTime for w in interval_words[i_w:next_i_w])
                aligned_tokens.append({
                    'text': ''.join(tokens[i_t:next_i_t]).strip(),
                    'audio_span': (interval_start, interval_end),
                })
                i_w = next_i_w
                i_t = next_i_t
                continue
            else:
                interval_start = min(w.minTime for w in interval_words[i_w:])
                interval_end = max(w.maxTime for w in interval_words[i_w:])
                aligned_tokens.append({
                    'text': tokens[i_t].strip(),
                    'audio_span': (interval_start, interval_end),
                })
                i_t += 1
                i_w = len(interval_words)
                continue
        if i_t < len(tokens):
            aligned_tokens.append({
                'text': ''.join(tokens[i_t:]).strip(),
                'audio_span': (-1,-1),
            })
    return aligned_tokens


def main(videos, sub_folder, aligned_folder, videos_sorted=None, save_to=None):
    aligned_videos = {}
    for video_index, video_name in enumerate(tqdm(videos)):
        if os.path.isfile(os.path.join(save_to, video_name + '.json')):
            continue
        video_index = video_index if videos_sorted is None else videos_sorted.index(video_name)
        subtitles = json.load(open(os.path.join(sub_folder, video_name + '.json')))
        subtitles = norm_subtitles_spans(subtitles)

        textgrid_fp = os.path.join(aligned_folder, f"{video_index}.TextGrid")
        tg_aligned = textgrid.TextGrid()
        tg_aligned.read(textgrid_fp)

        aligned_tokens = align_tokens_in_subtitles(subtitles, tg_aligned)
        segmented_phrases = split_tokens_into_phrases(aligned_tokens)
        aligned_videos[video_name] = segmented_phrases
        if save_to is not None:
            json.dump(segmented_phrases, open(os.path.join(save_to, video_name + '.json'), 'w'), ensure_ascii=False)
    return aligned_videos


if __name__ == '__main__':
    standup_root = '../standup_eng'
    aligned_folder = os.path.join(standup_root, 'mfa_data/standup_eng_aligned_beam100_retry_beam400')
    segmented_folder = os.path.join(standup_root, 'subtitles_faligned')
    sub_folder = os.path.join(standup_root, 'sub_postproc')
    metadata = json.load(open(os.path.join(standup_root, 'meta_data.json'), encoding='utf-8'))
    videos = sorted(list(metadata.keys()))
    aligned_videos = main(videos, sub_folder, aligned_folder, save_to=segmented_folder)
