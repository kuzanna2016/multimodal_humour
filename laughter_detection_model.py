import os, sys, librosa, torch, numpy as np

sys.path.append('./utils/')
import laugh_segmenter
import models, configs
import dataset_utils, audio_utils, data_loaders, torch_utils
from functools import partial
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def set_up_ld_model():
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
    return model, feature_fn, sample_rate, config, device


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
