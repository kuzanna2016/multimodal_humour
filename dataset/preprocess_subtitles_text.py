import os
import re
import argparse
import json
from tqdm import tqdm
from utils import clean_title
from numeric import ordinal_endings

REPLACED_SWEARS_DICT = json.load(open('replaced_swears.json'))

parser = argparse.ArgumentParser()

parser.add_argument("--dataset_root", type=str, default='../standup_dataset', help="Path to the dataset folder")
parser.add_argument("--lang", type=str, default='ENG', help="RUS or ENG dataset to preprocess")

def clean_up_whitespace(s):
    s = re.sub(r'\s+', r' ', s)
    return s.strip()


# ==============================================================================================================
def preproc_rus(text):
    text = clean_up_whitespace(text)
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
    return text


# ==============================================================================================================


def main(args):
    postprocessed_sub_folder = os.path.join(args.dataset_root, 'preprocessed_sub', 'subtitles_cleaned')
    os.makedirs(postprocessed_sub_folder, exist_ok=True)
    for fn in tqdm(os.listdir(os.path.join(args.dataset_root, 'sub'))):
        video_name = clean_title(os.path.splitext(fn)[0])
        subtitles = json.load(open(os.path.join(args.dataset_root, 'sub', fn)))
        if args.lang == 'RUS':
            subtitles = [[*s[:2], preproc_rus(s[2])] for s in subtitles if s[0] != s[1]]
        elif args.lang == 'ENG':
            subtitles = [[*s[:2], preproc_eng(s[2], video_name)] for s in subtitles if s[0] != s[1]]
        subtitles = [s for s in subtitles if s[2]]
        json.dump(subtitles, open(os.path.join(postprocessed_sub_folder, fn), 'w'), ensure_ascii=False)


if __name__ == '__main__':
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
