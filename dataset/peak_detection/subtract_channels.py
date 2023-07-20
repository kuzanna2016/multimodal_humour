import os
import json
import argparse
from tqdm import tqdm
import librosa
import soundfile as sf

from utils import clean_title
from const import PEAK_DETECTION_VIDEOS

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_root", type=str, default='../standup_dataset', help="Path to the dataset folder")
parser.add_argument("--all_videos", action="store_true",
                    help="Whether to subtract channels in all audio folder files, by default only considers audios picked for peak detection")


def main(args):
    meta_data = json.load(open(os.path.join(args.dataset_root, 'meta_data.json')))
    if args.all_videos:
        videos = meta_data.keys()
    else:
        videos = [clean_title(v) for vs in PEAK_DETECTION_VIDEOS.values() for v in vs]
    audio_folder = os.path.join(args.dataset_root, 'audio')
    audio_folder_wav = os.path.join(args.dataset_root, 'audio_subtracted')
    os.makedirs(audio_folder_wav, exist_ok=True)

    for video_name in tqdm(videos):
        fn = video_name + '.mp4'
        audio_path_mp4 = os.path.join(audio_folder, fn)
        audio_path_wav = os.path.join(audio_folder_wav, video_name + '.wav')
        x, sr = librosa.load(audio_path_mp4, sr=None, mono=False)
        sub = x[0, :] - x[1, :]
        sf.write(audio_path_wav, sub, sr)


if __name__ == '__main__':
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
