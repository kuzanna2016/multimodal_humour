from transformers import AutoProcessor, ASTModel
import torch
import librosa
import numpy as np

AST_SR = 16000


def load_model():
    processor = AutoProcessor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
    model = ASTModel.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    return processor, model, device


def get_features(audio_regions, region, device, model, processor, trim=0, batch_size=24):
    regions = []
    for i, (start, end) in enumerate(audio_regions):
        if trim > 0:
            end = start + trim if end - start > trim else end
        cut_region = region.seconds[start:end]
        sample = cut_region.samples
        resampled_region = librosa.resample(librosa.to_mono(sample), orig_sr=region.sr, target_sr=AST_SR)
        regions.append(resampled_region)
    n_regions = len(regions)
    inputs = processor(regions, sampling_rate=AST_SR, return_tensors="pt")
    input_values = inputs['input_values'].to(device)

    batch_outputs = []
    for i in range(int(np.ceil(n_regions / batch_size))):
        end = (i + 1) * batch_size
        end = end if end < n_regions else n_regions
        with torch.no_grad():
            outputs = model(input_values=input_values[i * batch_size:end])
        last_hidden_states = outputs.last_hidden_state
        bs = last_hidden_states.shape[0]
        last_hidden_states = last_hidden_states.cpu().numpy()
        last_hidden_states = last_hidden_states.reshape(bs, -1)
        batch_outputs.append(last_hidden_states)
    last_hidden_states = np.vstack(batch_outputs)
    return last_hidden_states
