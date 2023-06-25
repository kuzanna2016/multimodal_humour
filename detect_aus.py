import os
from feat import Detector
from tqdm.notebook import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def main(folder_with_videos, output_folder):
    detector = Detector(device="cuda:0")
    # Loop over and process each video and save results to csv
    for video in tqdm(os.listdir(folder_with_videos)):
        video_fp = os.path.join(folder_with_videos, video)
        out_name = os.path.join(output_folder, video_fp.replace(".mp4", ".csv"))
        if not os.path.exists(out_name):
            print(f"Processing: {video}")

            # This is the line that does detection!
            fex = detector.detect_video(video_fp)

            fex.to_csv(out_name, index=False)

if __name__ == '__main__':
    standup_root = '../standup_eng'
    video_folder = os.path.join(standup_root, 'video')
    output_folder = os.path.join(standup_root, 'OpenFace')
    os.makedirs(output_folder, exist_ok=True)
    main(video_folder, output_folder)