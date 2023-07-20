import os
import argparse
import numpy as np
import json
import pandas as pd
from tqdm import tqdm

from utils import get_documents, get_splits_audio_spans_labels

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_root", type=str, default='../standup_dataset', help="Path to the dataset folder")


def prepare_df(of_df):
    of_df.columns = [col.replace(" ", "") for col in of_df.columns]
    face_parameters_index = of_df.loc[:, 'p_0':'p_33'].columns.values
    aus_index = of_df.loc[:, 'AU01_r':'AU45_c'].columns.values
    gaze_index = ['gaze_angle_x', 'gaze_angle_y']
    columns = ['timestamp', *gaze_index, *aus_index, *face_parameters_index]
    of_df = of_df.loc[of_df.confidence >= 0.8, columns]
    of_df['timestamp'] = pd.to_datetime(of_df['timestamp'], unit='s')
    of_df.set_index('timestamp', inplace=True)
    of_df = of_df.fillna(0)
    return of_df


def get_span(df, span):
    start = pd.to_datetime(span[0], unit='s')
    end = pd.to_datetime(span[1], unit='s')
    filtered_df = df[(df.index >= start) & (df.index <= end)]
    return filtered_df.to_numpy()


def main(args, window=5):
    meta_data = json.load(open(os.path.join(args.dataset_root, 'meta_data.json')))
    documents = get_documents(args.dataset_root)
    _, split_labels = get_splits_audio_spans_labels(documents, sorted(meta_data.keys()))

    open_face_features = np.empty((len(split_labels), 71 * 2), dtype=float)
    open_face_folder = os.path.join(args.dataset_root, 'features', 'openface_features')
    i = 0
    for video_name in tqdm(sorted(meta_data.keys())):
        subs = documents[video_name]
        of_df = pd.read_csv(os.path.join(open_face_folder, video_name + '.csv'))
        of_df = prepare_df(of_df)

        for split in zip(*[subs[i:] for i in range(window)]):
            context_span = split[0]['audio_span'][0], split[3]['audio_span'][1]
            utterance_span = split[-1]['audio_span']

            context_features = np.nan_to_num(get_span(of_df, context_span)).mean(axis=0)
            utterance_features = np.nan_to_num(get_span(of_df, utterance_span)).mean(axis=0)
            open_face_features[i] = np.hstack([context_features, utterance_features])
            i += 1

    os.makedirs(os.path.join(args.dataset_root, 'features'), exist_ok=True)
    np.save(os.path.join(args.dataset_root, 'features', 'facial_mean_context_utterance_features.npy'),
            open_face_features)


if __name__ == '__main__':
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
