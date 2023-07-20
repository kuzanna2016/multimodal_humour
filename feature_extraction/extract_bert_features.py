import os
import argparse
import torch

from models.bert import load_model, extract_features, create_dataset as create_bert_dataset
from models.text_train_validate import create_dataset

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_root", type=str, default='../standup_dataset', help="Path to the dataset folder")
parser.add_argument("--model_name", type=str, default='svm',
                    help="One of the model names [bert, rubert, bert_multilingual]")


def main(args):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dataset = create_dataset(args.dataset_root)
    model, tokenizer = load_model(args.model_name, device, with_classification_head=False)
    dataset = create_bert_dataset(dataset, tokenizer, device)

    tokens_fp = os.path.join(args.dataset_root, 'features', 'bert_features', 'tokens.jsonl')
    embeddings_fp = os.path.join(args.dataset_root, 'features', 'bert_features', 'embeddings.npy')
    extract_features(dataset, model, tokenizer, tokens_fp, embeddings_fp)


if __name__ == '__main__':
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
