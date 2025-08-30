'''Extract segments of interest using ffmpeg. Use copy option in cmd to save time and space due to Colab resource limitations.'''
import os
import json
import subprocess
import argparse

def extract_clips(queries_file, video_dir, clips_dir):
    os.makedirs(clips_dir, exist_ok=True)

    # Load the predictions/queries from JSON
    with open(queries_file, "r") as f:
        queries = json.load(f)

    for idx, item in enumerate(queries):
        uid = item["video_uid"]
        start, end = item["pred"]  # [start_time, end_time] in seconds
        duration = end - start

        input_video = os.path.join(video_dir, f"{uid}.mp4")
        output_path = os.path.join(clips_dir, f"{uid}_clip_{idx:02d}.mp4")

        if not os.path.exists(input_video):
            print(f"Missing video: {uid}")
            continue

        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-ss", str(start), "-i", input_video, "-t", str(duration),
            "-c", "copy", output_path
        ]

        try:
            subprocess.run(cmd, check=True)
            print(f"Extracted: {output_path}")
        except subprocess.CalledProcessError:
            print(f"Error extracting {uid}, clip {idx:02d}")

    print("Extraction completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract clips from videos based on predictions/queries"
    )
    parser.add_argument("--queries_file", type=str, required=True,
                        help="JSON file containing predictions or queries")
    parser.add_argument("--video_dir", type=str, required=True,
                        help="Directory containing full videos")
    parser.add_argument("--clips_dir", type=str, required=True,
                        help="Directory where extracted clips will be saved")
    args = parser.parse_args()

    extract_clips(args.queries_file, args.video_dir, args.clips_dir)

