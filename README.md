# Multimodal Humour detection in stand-up comedy
Materials from the paper "Multimodal and Multilingual Laughter Detection in Stand-Up Comedy Videos"

## Dataset Collection, Preprocessing, and Laughter Detection Labeling

The scripts for dataset collection, preprocessing, and laughter detection labeling can be found in the `dataset` folder:

- `collect_videos.py`: Crawls and downloads videos, audios, and subtitles from YouTube.
- `preprocess_subtitles_text.py`: Preprocesses subtitle text by removing artifacts, cleaning up whitespace and
  punctuation, and reconstructing censored words.
- `prepare_for_mfa.py`: Prepares TextGrid files for forced alignment with MFA, including converting numbers and
  characters into full written form.
- `resegment_and_word_align.py`: Aligns forced aligned words with tokens and resegments subtitles.
- `extract_textual_laughter_markers.py`: Extracts spans of subtitles containing auditory laughter markers
  like `[audience laughs]`.
- `label_with_textual_laughter_markers.py`: Labels humor based on textual laughter markers.
- `swear_words_rus.py`: Regular expressions for replacing censored swear words.
- `numeric.py`: Auxiliary functions for working with number conversion.
- `laughter_detection`: Laughter detection experiments for the machine learning approach. Should be run inside the
  [laughter-detection project](https://github.com/jrgillick/laughter-detection) after installing its
  requirements:
    - `label_with_annotation.py`: Labels validation videos with manually annotated laughter.
    - `laughter_detection_model.py`: Sets up the laughter-detection model.
    - `laughter_detection_experiments.py`: Runs hyperparameter search for the laughter-detection model.
    - `label_with_laughter_detection.py`: Labels videos with laughter-detection results.
    - `vocal_remover.py`: Runs the vocal-remover model. Should be run from inside the [vocal-remover
      project](https://github.com/tsurumeso/vocal-remover) after installing its requirements.
- `peak_detection`: Laughter detection experiments for the peak detection approach:
    - `extract_audio_features.py`: Sets up the AudioTransformer model for audio features extraction.
    - `peak_detection_experiments.py`: Runs hyperparameter search for the peak detection threshold.
    - `clusterization_experiments.py`: Runs hyperparameter search for clusterization.
    - `plot_clustering.py`: Plots clustering results with PCA-reduced points.

## Feature Extraction for Multimodal SVM Model

The feature extraction scripts for the multimodal SVM model are in the `feature_extraction` folder:

- `extract_video_features.py`: Extracts video features using VideoMAE.
- `extract_bert_features.py`: Extracts textual features using BERT models.
- `extract_open_face_features.py`: Combines extracted OpenFace features.

## Humour Detection Models and Training Scripts

Humour detection models and their training scripts can be found in the `models` folder:

- `bert.py`: Contains BERT-based experiments.
- `colbert.py`: Includes ColBERT architecture and experiments.
- `svm.py`: Contains SVM experiments.

## Useful Plotting Scripts

Useful plotting scripts can be found in the `plotting` folder:

- `plot_audio_with_annotated_laughter.py`: Plots audio waveforms with annotated laughter and other spans (detected
  peaks, subtitle segmentation, word segmentation).
- `plot_dataset_statistics.py`: Plots dataset statistics such as the distribution of video duration, pauses before
  laughter, etc.

## Other Files

Other files include:

- `utils.py`: Contains data processing functions.
- `requirements.txt`: Lists the requirements to run the code.

