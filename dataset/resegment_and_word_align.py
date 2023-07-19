import re
import os
import json
from tqdm import tqdm
import argparse
from string import punctuation
import textgrid
from utils import get_tier_by_name, norm_subtitles_spans

punctuation += '»«…—'
punctuation = punctuation.replace(',', '')

TOKENS_SPLIT = r'\s+|\b[+=,!?:;"»«…/.—-]+\b'

parser = argparse.ArgumentParser()

parser.add_argument("--dataset_root", type=str, default='../standup_dataset', help="Path to the dataset folder")
parser.add_argument("--mfa_aligned_root", type=str, default='../standup_dataset', help="Path to the dataset folder")


def get_tokens(text):
    tokens = []
    pointer = 0
    i = None
    for i, m in enumerate(re.finditer(TOKENS_SPLIT, text, flags=re.IGNORECASE)):
        start, end = m.span(0)
        t = text[max(pointer - 1, 0):start]
        tokens.append(t)
        pointer = end
    if i is not None:
        tokens.append(text[pointer - 1:])
    else:
        tokens.append(text)
    return tokens


def garbage_clean(text):
    text = re.sub(r'[^\w\']', r'', text)
    text = re.sub(r'\s+', r' ', text.strip())
    text = text.strip().lower()
    return text


def add_new_phrase(phrases, current_tokens):
    for token in current_tokens:
        token.pop('isupper')
    valid_audio_spans = [t['audio_span'] for t in current_tokens if t['audio_span'] != (-1, -1)]
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

        tokens = get_tokens(text)
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
                    'audio_span': (-1, -1),
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
                if ''.join(interval_strs[i_w:i_w + 2]) == tokens_clean[i_t]:
                    aligned_tokens.append({
                        'text': tokens[i_t].strip(),
                        'audio_span': (interval_words[i_w].minTime, interval_words[i_w + 1].maxTime),
                    })
                    i_t += 1
                    i_w += 2
                    continue
            if i_t + 1 < len(tokens):
                next_matches = [
                    (j_t, [j for j in range(i_w + 1, len(interval_words))
                           if tokens_clean[j_t] == interval_strs[j]])
                    for j_t in range(i_t + 1, len(tokens))
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
                'audio_span': (-1, -1),
            })
    return aligned_tokens


def align(videos, sub_folder, aligned_folder, videos_sorted=None, save_to=None):
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


def main(args):
    aligned_folder = args.mfa_aligned_root
    segmented_folder = os.path.join(args.dataset_root, 'subtitles_faligned')
    sub_folder = os.path.join(args.dataset_root, 'sub')
    metadata = json.load(open(os.path.join(args.dataset_root, 'meta_data.json'), encoding='utf-8'))
    videos = sorted(list(metadata.keys()))
    align(videos, sub_folder, aligned_folder, save_to=segmented_folder)


if __name__ == '__main__':
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
