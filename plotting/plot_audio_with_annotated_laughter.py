import os
import argparse
import matplotlib.pyplot as plt
import librosa
import json
from utils import get_laughs_from_annotation, load_validation_data

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_root", type=str, default='../standup_dataset', help="Path to the dataset folder")
parser.add_argument("--audio_type", type=str, default='orig',
                    help="Type of audio to use from [orig, subtracted, vocal-remover]")
parser.add_argument("--video_name", type=str, help="Video to plot")
parser.add_argument("--laughter", action="store_true", help="Plot annotated laughter")
parser.add_argument("--peaks", type=str, default='', help="Which peak detection folder to plot, if none leave empty")


def subtract_channels(audio_path):
    x, sr = librosa.load(audio_path, sr=None, mono=False)
    sub = x[0, :] - x[1, :]
    return sub, sr


def main(args):
    audio_folder = os.path.join(args.dataset_root, 'audio')
    video_name = args.video_name
    titles, annotations = load_validation_data(args.dataset_root)
    if args.audio_type == 'orig':
        fp = os.path.join(audio_folder, video_name + '.mp4')
        data, sr = librosa.load(fp)
    elif args.audio_type == 'subtracted':
        fp = os.path.join(audio_folder, video_name + '.mp4')
        data, sr = subtract_channels(fp)
    else:
        audio_folder = os.path.join(args.dataset_root, 'vocal-remover')
        fp = os.path.join(audio_folder, video_name + '_Instruments.wav')
        data, sr = librosa.load(fp)
    data = librosa.to_mono(data)

    fig, ax = plt.subplots(1, 1, figsize=(30, 5))
    ax.plot(data, alpha=0.9)
    if args.laughter:
        true_detections = get_laughs_from_annotation(annotations, video_name, include_applause=False)
        for s, e in true_detections:
            s, e = s * sr, e * sr
            ax.axvspan(s, e, fill=True, color='g', alpha=0.2)
    if args.peaks:
        spans_folder = os.path.join(args.dataset_root, 'experiments','detected_peaks')
        audio_regions = json.load(open(os.path.join(spans_folder, args.peaks, video_name + '.json')))
        for s, e in audio_regions:
            s, e = s * sr, e * sr
            ax.axvspan(s, e, fill=True, color='r', alpha=0.2)
    ax.set_title(video_name)
    name_values = [
        'laughter' if args.laughter else '',
        args.peaks,
        args.audio_type
    ]
    name = video_name + '_' + '_'.join([v for v in name_values if v])
    os.makedirs(os.path.join(args.dataset_root, 'plots'), exist_ok=True)
    print('Saving to', os.path.join(args.dataset_root, 'plots', name + '.png'))
    fig.savefig(os.path.join(args.dataset_root, 'plots', name + '.png'))


if __name__ == '__main__':
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
