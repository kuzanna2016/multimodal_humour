import os
import json
import subprocess
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()

parser.add_argument("--dataset_root", type=str, default='../standup_dataset', help="Path to the dataset folder")


def main(args):
    audio_folder = os.path.join(args.dataset_root, 'audio')
    meta_data = json.load(open(os.path.join(args.dataset_root, 'meta_data.json')))
    temp_fp = os.path.join(args.dataset_root, 'temp.mp4')

    output_folder = os.path.join(args.dataset_root, 'vocal-remover')
    os.makedirs(output_folder, exist_ok=True)

    for video_name in tqdm(sorted(meta_data.keys())):
        fp = os.path.join(audio_folder, video_name + '.mp4')
        if os.path.isfile(os.path.join(output_folder, video_name + '_Instruments.wav')):
            continue
        rv = subprocess.run(
            f'ffmpeg -i "{fp}" -c copy "{temp_fp}"  -hide_banner -loglevel error',
            shell=True, check=True, text=True)
        fp = temp_fp
        rv = subprocess.run(
            f'python inference.py --input "{fp}" --gpu 1 --output_dir "{output_folder}"',
            shell=True, check=True, text=True)
        os.remove(temp_fp)
        os.remove(os.path.join(output_folder, 'temp_Vocals.wav'))
        os.rename(os.path.join(output_folder, 'temp_Instruments.wav'),
                  os.path.join(output_folder, video_name + '_Instruments.wav'))

if __name__ == '__main__':
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)