from transformers import VideoMAEImageProcessor, VideoMAEModel
import numpy as np
import torch
import cv2, os
import json
from tqdm import tqdm
from math import ceil
from utils import get_documents
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_root", type=str, default='../standup_dataset', help="Path to the dataset folder")
parser.add_argument("--bs", type=int, default=32, help="Batch size")


def get_all_frames(video, start=0, end=None, frame_rate=5):
    video.set(cv2.CAP_PROP_POS_MSEC, start)
    frames = []
    timestamps = []
    length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video.get(cv2.CAP_PROP_FPS)
    for i in tqdm(range(length), total=length if end is None else ceil((end - start) / 1000 * fps)):
        # Read the frame
        success, frame = video.read()
        ms = video.get(cv2.CAP_PROP_POS_MSEC)
        if end is not None and ms > end:
            return frames, timestamps
        if success:
            if i % frame_rate == 0:
                frames.append(frame)
                timestamps.append(ms)
        else:
            break
    return frames, timestamps


def get_sample(frames, timestamps, start_time, end_time, size=16):
    # Convert time to milliseconds
    start_ms = start_time * 1000
    end_ms = end_time * 1000

    frames = [frame for i, frame in zip(timestamps, frames) if start_ms <= i <= end_ms]
    if len(frames) != size:
        indices = np.linspace(0, len(frames), num=size)
        indices = np.clip(indices, 0, len(frames) - 1).astype(np.int64)
        frames = [frames[i] for i in indices]
    return frames


def get_features(batch_frames, processor, model, device):
    inputs = processor(batch_frames, return_tensors="pt")
    with torch.no_grad():
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = model(**inputs)
        outputs = outputs.last_hidden_state.cpu().numpy()
        return outputs


def main(args, window=5):
    meta_data = json.load(open(os.path.join(args.dataset_root, 'meta_data.json')))
    documents = get_documents(args.dataset_root)
    video_folder = os.path.join(args.dataset_root, 'video')

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    processor = VideoMAEImageProcessor.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics")
    model = VideoMAEModel.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics").to(device)
    output_folder = os.path.join(args.dataset_root, 'features', 'video_features')
    os.makedirs(output_folder, exist_ok=True)

    for i, video_name in enumerate(sorted(meta_data.keys())):
        subs = documents[video_name]
        video = cv2.VideoCapture(os.path.join(video_folder, video_name + '.mp4'))
        n_splits = len(subs) - window + 1
        features = np.empty((n_splits, 1568, 768), dtype=np.float32)

        splits = list(zip(*[subs[i:] for i in range(window)]))
        for j in range(ceil(n_splits / args.bs)):
            batch_splits = splits[j * args.bs:(j + 1) * args.bs]
            batch_start = batch_splits[0][0]['audio_span'][0]
            batch_end = batch_splits[-1][-1]['audio_span'][1]
            batch_frames, timestamps = get_all_frames(video, start=batch_start * 1000, end=batch_end * 1000)
            batch_sample = []
            for split in tqdm(batch_splits):
                start_c = split[0]['audio_span'][0]
                end_c = split[-1]['audio_span'][1]
                frames = get_sample(batch_frames, timestamps, start_c, end_c)
                batch_sample.append(frames)
            outputs = get_features(batch_sample, processor, model, device)
            features[j * args.bs:min((j + 1) * args.bs, n_splits)] = outputs
        np.save(os.path.join(output_folder, video_name + '.npy'), features)


if __name__ == '__main__':
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
