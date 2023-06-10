import os, sys, librosa, torch, numpy as np, pandas as pd
import subprocess

sys.path.append('./utils/')
import laugh_segmenter
import models, configs
import dataset_utils, audio_utils, data_loaders, torch_utils
from tqdm import tqdm
from functools import partial

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

sample_rate = 8000
model_path = 'checkpoints/in_use/resnet_with_augmentation'
config = configs.CONFIG_MAP['resnet_with_augmentation']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device {device}")

model = config['model'](dropout_rate=0.0, linear_layer_size=config['linear_layer_size'],
                        filter_sizes=config['filter_sizes'])
feature_fn = config['feature_fn']
model.set_device(device)

if os.path.exists(model_path):
    torch_utils.load_checkpoint(model_path + '/best.pth.tar', model)
    model.eval()
else:
    raise Exception(f"Model checkpoint not found at {model_path}")


def load(audio_path, feature_fn, sample_rate, config):
    ##### Load the audio file and features

    inference_dataset = data_loaders.SwitchBoardLaughterInferenceDataset(
        audio_path=audio_path, feature_fn=feature_fn, sr=sample_rate)

    collate_fn = partial(audio_utils.pad_sequences_with_labels,
                         expand_channel_dim=config['expand_channel_dim'])

    inference_generator = torch.utils.data.DataLoader(
        inference_dataset, num_workers=4, batch_size=8, shuffle=False, collate_fn=collate_fn)
    return inference_generator


def predict(inference_generator, device):
    ##### Make Predictions

    probs = []
    for model_inputs, _ in inference_generator:
        x = torch.from_numpy(model_inputs).float().to(device)
        preds = model(x).cpu().detach().numpy().squeeze()
        if len(preds.shape) == 0:
            preds = [float(preds)]
        else:
            preds = list(preds)
        probs += preds
    probs = np.array(probs)
    return probs


def cut_threshold(probs, threshold, min_length, audio_path, log=1):
    file_length = audio_utils.get_audio_length(audio_path)
    fps = len(probs) / float(file_length)
    probs = laugh_segmenter.lowpass(probs)
    instances = laugh_segmenter.get_laughter_instances(probs, threshold=threshold, min_length=float(min_length),
                                                       fps=fps)
    if log > 0:
        print();
        print("found %d laughs." % (len(instances)))
    return instances


def save_predictions(instances, video_name, output_dir):
    os.system(f'mkdir -p "{output_dir}"')
    with open(os.path.join(output_dir, video_name + '.tsv'), 'w') as f:
        f.write('start\tend\tduration\n')
        for i in instances:
            f.write(f'{i[0]}\t{i[1]}\t{i[1] - i[0]}\n')
    print('Saved laughter segments in {}'.format(
        os.path.join(output_dir, video_name + '.tsv')))


import re
import json
from collections import defaultdict
from sklearn.metrics import precision_recall_fscore_support


def clean_title(title):
    title = re.sub(r'[|.,"/]', r'', title)
    title = re.sub(r'й', r'й', title)
    title = re.sub(r'ё', r'ё', title)
    return title


standup_root = '/data/disk1/share/akuznetsova/standup_rus'
audio_folder = os.path.join(standup_root, 'audio')
FA_SUBTITLES_FOLDER = os.path.join(standup_root, 'subtitles_faligned')
with open(os.path.join(standup_root, 'annotation', 'titles.txt'), encoding='utf-8') as f:
    titles = f.read().splitlines()
titles = [clean_title(t) for t in titles]

dfs = [
    pd.read_csv(os.path.join(standup_root, 'annotation', f'standup{i}.txt'),
                sep='\t',
                names=['tier', 'start', 'end', 'duration'],
                usecols=[0, 2, 3, 4], dtype={'tier': str, 'start': float, 'end': float, 'duration': float})
    for i in range(len(titles))
]
for i, df in enumerate(dfs):
    df['video_name'] = titles[i]

annotations = pd.concat(dfs, axis=0)
meta_data = json.load(open(os.path.join(standup_root, 'meta_data.json')))


def cut_segment(fp, start, end, temp_fp=standup_root, play=True, rm=True, ext='mp4', with_codec=False):
    duration = end - start
    temp_fp = os.path.join(temp_fp, f'temp.{ext}')
    if with_codec:
        rv = subprocess.run(
            f'ffmpeg -ss {start} -i "{fp}" -to {duration} -c:a aac "{temp_fp}"  -hide_banner -loglevel error',
            shell=True, check=True, text=True)
    else:
        rv = subprocess.run(
            f'ffmpeg -ss {start} -i "{fp}" -to {duration} -c copy "{temp_fp}"  -hide_banner -loglevel error',
            shell=True, check=True, text=True)
    if rm:
        os.remove(temp_fp)


def get_laughs_from_annotation(annotations, video_name, include_applause=False):
    if include_applause:
        mask = annotations.video_name == video_name
    else:
        mask = (annotations.video_name == video_name) & (annotations.tier == 'laughter')
    true_laughs = annotations[mask]
    laughs = [(r['start'], r['end']) for r in true_laughs.to_dict('records')]
    return laughs


def interval_overlap(interval0, interval1):
    overlap = max([
        0,
        min([interval0[1], interval1[1]]) - max([interval0[0], interval1[0]])
    ])
    return overlap


def detect_laughs_in_subtitle(laughs, start, end):
    for s, e in laughs:
        if interval_overlap((s, e), (start, end)) > 0:
            yield (s, e)


def get_search_windows(subtitles, max_duration, max_window_length=0.7, min_pause_length=0.2):
    for ((_, end, _), (start, _, _)) in zip(subtitles, subtitles[1:]):
        duration = start - end
        if duration > min_pause_length:
            yield (end, start)
        else:
            yield (end, min([end + max_window_length, max_duration]))
    end = subtitles[-1][1]
    yield (end, min([end + max_window_length, max_duration]))

temp_fp = os.path.join(standup_root, 'temp0.mp4')
log = defaultdict(lambda: defaultdict(list))

for condition in ['realigned', 'vocal-remover']:
    for video_name in titles:
        if condition == 'realigned':
            audio_fp = os.path.join(audio_folder, video_name + '.mp4')
        else:
            audio_fp = os.path.join(standup_root, 'vocal-remover', video_name + '_Instruments.wav')
        subtitles_fp = os.path.join(standup_root, 'subtitles_faligned_annotation_labeled', video_name + '.json')
        annotated_subtitles = json.load(open(subtitles_fp))
        subtitles = [s[:3] for s in annotated_subtitles]

        pred = defaultdict(list)
        true = [s[3] for s in annotated_subtitles]
        laughs = get_laughs_from_annotation(annotations, video_name)
        duration = meta_data[video_name].get('duration', subtitles[-1][1])
        search_windows = list(
            get_search_windows(subtitles, max_duration=duration))
        if condition == 'vocal-remover':
            subprocess.run(
                f'ffmpeg -i "{audio_fp}" -c:a aac "{temp_fp}"  -hide_banner -loglevel error',
                shell=True, check=True, text=True)
            audio_fp = temp_fp
        inference_generator = load(audio_fp, feature_fn, sample_rate, config)
        probs = predict(inference_generator, device)
        for threshold in [0.1, 0.2, 0.3, 0.4, 0.5]:
            for min_length in [0.01, 0.05, 0.1, 0.15, 0.2]:
                instances = cut_threshold(probs, threshold, min_length, audio_fp, log=0)
                pred_laughter = [
                    any(interval_overlap((s, e), (cut_start, cut_end)) > 0 for s, e in instances)
                    for (cut_start, cut_end) in search_windows
                ]
                pred[f'th{threshold}_ml{min_length}_{condition}'] = pred_laughter
        if condition == 'vocal-remover':
            os.remove(temp_fp)
        for k, predictions in pred.items():
            precision, recall, f1, _ = precision_recall_fscore_support(true, predictions, average='binary')
            log[k]['precision'].append(precision)
            log[k]['recall'].append(recall)
            log[k]['f1'].append(f1)
json.dump(log, open(os.path.join(standup_root, 'laughter_detection_in_subtitle_whole_cross_validation.json'), 'w'))
