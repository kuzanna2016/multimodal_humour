import os
import subprocess
from time import sleep
from collections import defaultdict
from tqdm.notebook import tqdm
from utils import clean_title
import json
import argparse

from pytube import Channel
from pytube import YouTube
from langdetect import detect
from youtube_transcript_api import YouTubeTranscriptApi

channels = {
    'RUS': [
        'OUTSIDESTANDUP',
        'standupedwingroup',
        'denis_chuzhoy',
        'schastlivcy',
        'user-ff9dx9iy5f',
        'orlov_skvz',
        'VSESVOII',
        'SobolevTUT',
        'molchanrecord',
        'IonOFF24'],
    'ENG': [
        'standup'
    ]
}

TRANSCRIPT_LANGS = {
    'RUS': ['ru'],
    'ENG': ['en-US', 'en-UK', 'en']
}

KWS = [
    'stand up',
    'standup',
    'stand-up',
    'стендап',
    'стенд ап',
    'концерт',
    'full special'
]

parser = argparse.ArgumentParser()

parser.add_argument("--lang", type=str, default='ENG', help="RUS or ENG dataset to download")
parser.add_argument("--dataset_root", type=str, default='../standup_dataset', help="Path to the output folder")


def get_title(yt):
    while True:
        try:
            title = yt.title
            break
        except:
            sleep(1)
            yt = YouTube(yt.watch_url)
            continue
    return title


def get_videos(lang):
    videos_with_captions = defaultdict(list)
    cumulative_time = 0
    max_time = 72000
    for name in channels[lang]:
        print('Channel', name)
        c = Channel(f'https://www.youtube.com/@{name}')
        try:
            for video in tqdm(c.videos):
                if cumulative_time > max_time:
                    break
                try:
                    title = get_title(video)
                    if not any(kw in title.lower() for kw in KWS):
                        continue
                    transcript_list = YouTubeTranscriptApi.list_transcripts(video.video_id)
                    transcript = transcript_list.find_manually_created_transcript(TRANSCRIPT_LANGS[lang])
                    captions = transcript.fetch()
                    if not captions:
                        continue
                    captions_text = ' '.join([s['text'] for s in captions])
                    caption_lang = detect(captions_text)
                    print('Detected', caption_lang)
                    if caption_lang == lang[:2].lower():
                        print('Video is a match', title)
                        videos_with_captions[name].append({
                            'url': video.watch_url,
                            'title': clean_title(title),
                            'video': video,
                            'captions': captions,
                        })
                        cumulative_time += video.length
                except Exception as e:
                    print('Problem with video')
                    print(e)
                    continue
        except Exception as e:
            print('Not found channel', name)
            print(e)
            continue
    return videos_with_captions


def download_media(video, video_path, audio_path):
    title = video['title']
    url = video['url']
    video_output_path = os.path.join(video_path, title + '.mp4')
    subprocess.run(
        f'youtube-dl -f "bestvideo[ext=mp4]" -o "{video_output_path}" {url}',
        shell=True, check=True, text=True)

    audio_output_path = os.path.join(audio_path, title + '.%(ext)s')
    subprocess.run(
        f'youtube-dl -f "bestaudio" -o "{audio_output_path}" {url}',
        shell=True, check=True, text=True)


def download_captions(video, sub_path):
    title = video['title']
    captions = video['captions']
    captions = [[c['start'], c['start'] + c['duration'], c['text']] for c in captions]
    json.dump(captions, open(os.path.join(sub_path, title + '.json'), 'w'), ensure_ascii=False)

def convert_audio_to_mp4(audio_path):
    for fn in os.listdir(audio_path):
        video_name, ext = os.path.splitext(fn)
        if ext == '.mp4':
            continue
        audio_fp = os.path.join(audio_path, fn)
        audio_no_ext_fp = os.path.join(audio_path, video_name)
        subprocess.run(
            f'ffmpeg -i "{audio_fp}" "{audio_no_ext_fp}.mp4"  -hide_banner -loglevel error',
            shell=True, check=True, text=True)
        os.remove(audio_fp)

def main(args):
    videos_with_captions = get_videos(args.lang)
    metadata = {
        clean_title(v['title']): {'channel': k, 'url': v['url'], 'duration': v['video'].length}
        for k, values in videos_with_captions.items()
        for v in values
    }
    json.dump(metadata, open(os.path.join(args.dataset_root, 'meta_data.json'), 'w', encoding='utf-8'),
              ensure_ascii=False)

    sub_path = os.path.join(args.dataset_root, 'sub')
    os.makedirs(sub_path, exist_ok=True)
    video_path = os.path.join(args.dataset_root, 'video')
    os.makedirs(video_path, exist_ok=True)
    audio_path = os.path.join(args.dataset_root, 'audio')
    os.makedirs(audio_path, exist_ok=True)

    for videos in videos_with_captions.values():
        for v in videos:
            download_captions(v, sub_path)
            download_media(v, video_path, audio_path)
    convert_audio_to_mp4(audio_path)


if __name__ == '__main__':
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
