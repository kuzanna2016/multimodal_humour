import re
import json
import argparse
import os

LAUGHTER_REGEXP = re.compile(r'[([].*?(?:laugh|chuckl|giggl|whoop).*?[])]', flags=re.IGNORECASE|re.DOTALL)

parser = argparse.ArgumentParser()

parser.add_argument("--dataset_root", type=str, default='../standup_dataset', help="Path to the dataset folder")


def main(args):
    extracted_laughter_from_sub_folder = os.path.join(args.dataset_root, 'extracted_laughter_from_sub')
    os.makedirs(extracted_laughter_from_sub_folder, exist_ok=True)

    for fn in os.listdir(os.path.join(args.dataset_root, 'sub')):
      video_name = os.path.splitext(fn)[0]
      subtitles = json.load(open(os.path.join(args.dataset_root, 'sub', fn)))
      laughs = [
          [start, end, s]
          for start, end, s in subtitles
          if LAUGHTER_REGEXP.search(s) is not None
      ]
      json.dump(laughs, open(os.path.join(extracted_laughter_from_sub_folder, fn), 'w'), ensure_ascii=False)

if __name__ == '__main__':
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)