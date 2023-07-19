from transformers import VideoMAEImageProcessor, VideoMAEModel
import numpy as np
import torch
import cv2, os
import json
from tqdm import tqdm
from math import ceil
from utils import clean_title

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
standup_root = '/data/disk1/share/akuznetsova/standup_rus'
video_folder = os.path.join(standup_root, 'video')

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(device)
processor = VideoMAEImageProcessor.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics")
model = VideoMAEModel.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics").to(device)
meta_data = json.load(open(os.path.join(standup_root, 'meta_data.json')))
subtitles_annotated_folder = os.path.join(standup_root, 'subtitles_faligned')
documents = {}
for fn in os.listdir(subtitles_annotated_folder):
    video_name = os.path.splitext(fn)[0]
    if video_name in documents:
        continue
    video_name = clean_title(video_name)
    subtitles = json.load(open(os.path.join(subtitles_annotated_folder, fn)))
    documents[video_name] = subtitles


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


def get_features(batch_frames):
    inputs = processor(batch_frames, return_tensors="pt")
    with torch.no_grad():
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = model(**inputs)
        outputs = outputs.last_hidden_state.cpu().numpy()
        return outputs


window = 5
bs = 32

output_folder = os.path.join(standup_root, 'dataset', 'video_features')
os.makedirs(output_folder, exist_ok=True)

for i, video_name in enumerate(sorted(meta_data.keys())):
    subs = documents[video_name]
    video = cv2.VideoCapture(os.path.join(video_folder, video_name + '.mp4'))
    batch_frames = []
    n_splits = len(subs) - window + 1
    features = np.empty((n_splits, 1568, 768), dtype=np.float32)

    splits = list(zip(*[subs[i:] for i in range(window)]))
    for j in range(ceil(n_splits / bs)):
        batch_splits = splits[j * bs:(j + 1) * bs]
        batch_start = batch_splits[0][0]['audio_span'][0]
        batch_end = batch_splits[-1][-1]['audio_span'][1]
        batch_frames, timestamps = get_all_frames(video, start=batch_start * 1000, end=batch_end * 1000)
        batch_sample = []
        for split in tqdm(batch_splits):
            # extract context segment
            start_c = split[0]['audio_span'][0]
            end_c = split[-1]['audio_span'][1]
            frames = get_sample(batch_frames, timestamps, start_c, end_c)
            batch_sample.append(frames)
        outputs = get_features(batch_sample)
        features[j * bs:min((j + 1) * bs, n_splits)] = outputs
    np.save(os.path.join(output_folder, video_name + '.npy'), features)
