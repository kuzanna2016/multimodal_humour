import os
import re
import textgrid
import argparse
import json
from tqdm import tqdm
import subprocess
from string import punctuation
from utils import norm_subtitles_spans, clean_title
from numeric import change_numeric_rus, NUMBER_REGEXP, change_numeric_eng
from swear_words_rus import REGEXPS as REGEXPS_RUS

parser = argparse.ArgumentParser()

parser.add_argument("--dataset_root", type=str, default='../standup_dataset', help="Path to the dataset folder")
parser.add_argument("--do_audio", action="store_true", help="Whether to also prepare audio")
parser.add_argument("--lang", type=str, default='ENG', help="RUS or ENG dataset to preprocess")


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
def prepare_rus(text):
    text = re.sub(r'(\d)%\s', r'\1 процентов ', text)
    text = re.sub(r'(\d)%-', r'\1 процент', text)
    text = re.sub(r'(\d)\s?ч\s', r'\1 часов ', text)
    text = re.sub(r'(\d)(час|лет|кг)', r'\1 \2', text)
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

def update_numbers(s):
    m = re.search(NUMBER_REGEXP, s)
    if m is not None:
        s = clean_up_whitespace(change_numeric_eng(s))
    return s


def prepare_eng(text):
    text = update_numbers(text)
    return text


# ==============================================================================================================


def main(args):
    corpus_folder = os.path.join(args.dataset_root, 'mfa_data')
    os.makedirs(corpus_folder, exist_ok=True)
    subtitles_folder = os.path.join(args.dataset_root, 'sub_postproc')
    metadata = json.load(open(os.path.join(args.dataset_root, 'meta_data.json'), encoding='utf-8'))

    videos = sorted(metadata.keys())
    for file in tqdm(os.listdir(os.path.join(subtitles_folder))):
        video_name = os.path.splitext(file)[0]
        video_name = clean_title(video_name)
        video_index = videos.index(video_name)
        textgrid_fp = os.path.join(corpus_folder, f"{video_index}.TextGrid")

        subtitles = json.load(open(os.path.join(subtitles_folder, file)))
        if args.lang == 'RUS':
            subtitles = [[*s[:2], prepare_rus(s[2])] for s in subtitles if s[0] != s[1]]
        elif args.lang == 'ENG':
            subtitles = [[*s[:2], prepare_eng(s[2])] for s in subtitles if s[0] != s[1]]
        subtitles = [s for s in subtitles if s[2]]
        convert_sentences_to_textgrid(subtitles, textgrid_fp, metadata[video_name]['channel'])

        audio_path = os.path.join(args.dataset_root, 'audio', video_name + '.mp4')
        audio_path_wav = os.path.join(corpus_folder, f'{video_index}.wav')
        if args.do_audio:
            subprocess.run(
                f'ffmpeg -i "{audio_path}" "{audio_path_wav}"  -hide_banner -loglevel error',
                shell=True, check=True, text=True)


if __name__ == '__main__':
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
