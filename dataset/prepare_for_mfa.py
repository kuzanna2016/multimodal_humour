import os
import re
import textgrid
import json
from tqdm import tqdm
import subprocess
from string import punctuation
from utils import norm_subtitles_spans, clean_title
from numeric import change_numeric_rus, ordinal_endings, NUMBER_REGEXP, change_numeric_eng
from swear_words_rus import REGEXPS as REGEXPS_RUS

REPLACED_SWEARS_DICT = json.load(open('replaced_swears.json'))


def convert_sentences_to_textgrid(sentences, output_file, speaker_name):
    tg = textgrid.TextGrid()

    # Create the interval tier
    interval_tier = textgrid.IntervalTier(name=speaker_name)
    sentences = norm_subtitles_spans(sentences)
    for sentence in sentences:
        start_time, end_time, text = sentence
        interval = textgrid.Interval(start_time, end_time, text)
        interval_tier.addInterval(interval)

    # Add the interval tier to the TextGrid
    tg.append(interval_tier)

    # Save the TextGrid to a file
    tg.write(output_file)


def preproc_swear_words_with_regexps(word, regexps):
    for reg, rep in regexps:
        word = re.sub(reg, rep, word, flags=re.IGNORECASE)
        if '*' not in word:
            return word
    return word


def clean_up_whitespace(s):
    s = re.sub(r'\s+', r' ', s)
    return s.strip()


# ==============================================================================================================
def preproc_rus(text):
    text = re.sub(r'\d+\s\d+:\d+:\d+,?\d*\s-->\s\d+:\d+:\d+,?\d*\s', r'', text, flags=re.IGNORECASE)
    text = re.sub(r'\d+:\d+:\d+:\d+\s-\s\d+:\d+:\d+:\d+\s(speaker\s\d+)?', r'', text, flags=re.IGNORECASE)
    text = re.sub(r'\[\s*__\s*\]', r'', text)
    text = re.sub(r'’', r"'", text)
    text = re.sub(r'(\d)%\s', r'\1 процентов ', text)
    text = re.sub(r'(\d)%-', r'\1 процент', text)
    text = re.sub(r'(\d)\s?ч\s', r'\1 часов ', text)
    text = re.sub(r'(\d)(час|лет|кг)', r'\1 \2', text)
    text = re.sub(r'(\w)-(\1-?){2,}', r'\1', text)
    text = re.sub(ordinal_endings, r'\1\2', text)
    words = re.split(r'\s+|\b[+=,!?:;"»«…/.—-]+\b', text, flags=re.IGNORECASE)

    new_words = []
    for w in words:
        w = re.sub(r'^[^\w*]+|[^*\w]+$', r'', w.strip())
        if '*' in w:
            w = preproc_swear_words_with_regexps(w, regexps=REGEXPS_RUS)
            w = re.sub(r'^[^\w]+|[^\w]+$', r'', w.strip())
        w = re.sub(r'(\w)([уыаоэ])\2+$', r'\1\2', w.strip(), flags=re.IGNORECASE)
        w = re.sub(r'^([уыаоэяию])\1+$', r'\1', w, flags=re.IGNORECASE)
        w = change_numeric_rus(w)
        if w in punctuation:
            continue
    text = ' '.join(new_words)
    return text


# ==============================================================================================================

def preproc_swear_words_with_dict(s, video_name):
    if "*" in s:
        replace = [r['replaced'] for r in REPLACED_SWEARS_DICT[video_name] if s == r['orig']][0]
        return replace
    else:
        return s


def clean_up_subtitle_turns_and_music(s):
    s = re.sub(r'♪', r'', s)
    s = re.sub(r'(?:- )?[([].*?[])]:?', r'', s)
    s = re.sub(r'\s+', r' ', s)
    return s.strip()


SPEAKER_REGEXP = re.compile(r'(?:- )?(?:audience|(?:male )?announcer|(?:wo)?man ?\d?|both|all|chris|zach)(?:>>|:)',
                            flags=re.IGNORECASE)


def clean_up_speaker_info(s):
    s = re.sub(r'♪', r'', s)
    s = re.sub(r'(?:- )?[([].*?[])]:?', r'', s)
    s = re.sub(r'\s+', r' ', s)
    return s.strip()


def update_numbers(s):
    m = re.search(NUMBER_REGEXP, s)
    if m is not None:
        s = clean_up_whitespace(change_numeric_eng(s))
    return s


def preproc_eng(text, video_name):
    text = clean_up_whitespace(text)
    text = preproc_swear_words_with_dict(text, video_name)
    text = clean_up_whitespace(text)
    if "captioned" in text.lower() or 'captioning' in text.lower():
        return ''
    text = clean_up_subtitle_turns_and_music(text)
    text = clean_up_whitespace(text)
    text = SPEAKER_REGEXP.sub(r'', text)
    text = clean_up_whitespace(text)
    if re.match(r'^\W+$', text) is not None:
        return ''
    text = update_numbers(text)
    return text


# ==============================================================================================================

PREPROCS = {
    'ENG': preproc_eng,
    'RUS': preproc_rus,
}


def main(args):
    corpus_folder = os.path.join(args.dataset_root, 'mfa_data')
    os.makedirs(corpus_folder, exist_ok=True)
    metadata = json.load(open(os.path.join(args.dataset_root, 'meta_data.json'), encoding='utf-8'))
    clean_text = PREPROCS.get(args.lang, lambda x: x)

    videos = sorted(metadata.keys())
    for file in tqdm(os.listdir(os.path.join(args.dataset_root, 'sub'))):
        video_name = os.path.splitext(file)[0]
        video_name = clean_title(video_name)
        video_index = videos.index(video_name)
        textgrid_fp = os.path.join(corpus_folder, f"{video_index}.TextGrid")

        subtitles = json.load(open(os.path.join(args.dataset_root, 'sub', file)))
        subtitles = [[*s[:2], clean_text(s[2])] for s in subtitles if s[0] != s[1]]
        subtitles = [s for s in subtitles if s[2]]
        convert_sentences_to_textgrid(subtitles, textgrid_fp, metadata[video_name]['channel'])

        audio_path = os.path.join(args.dataset_root, 'audio', video_name + '.mp4')
        audio_path_wav = os.path.join(corpus_folder, f'{video_index}.wav')
        if args.do_audio:
            subprocess.run(
                f'ffmpeg -i "{audio_path}" "{audio_path_wav}"  -hide_banner -loglevel error',
                shell=True, check=True, text=True)
