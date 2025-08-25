"""
Loads the Video-LLaVA model to generate answers to queries, by watching to video clips. 
For each video clip specified in an input JSON file, it extracts frames, sends the video 
and query to the model, and saves the generated answers in an output JSON file.
"""

import json
import os
import argparse
import av
import numpy as np
import torch
from transformers import VideoLlavaProcessor, VideoLlavaForConditionalGeneration, BitsAndBytesConfig
from tqdm import tqdm

def read_video_pyav(container, indices):
    frames = []
    container.seek(0)
    start_index=indices[0]
    end_index=indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])

def load_model():
    print("Starting Video-LLaVA model loading...")
    print("Model files will be downloaded")
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    model_name = "LanguageBind/Video-LLaVA-7B-hf"
    
    model = VideoLlavaForConditionalGeneration.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto"
    )
    
    processor = VideoLlavaProcessor.from_pretrained(model_name)
    
    print("Model loaded successfully!")
    return model, processor

def clean_answer(answer):
    """
    Removes only specific problematic characters: Ъ, trailing 'c', double periods, \n
    """
    # Remove Cyrillic character Ъ
    answer = answer.replace('Ъ', '')
    
    # Remove all \n and replace with space
    answer = answer.replace('\n', ' ')
    
    # Remove double or multiple periods at the end
    while answer.endswith('..'):
        answer = answer[:-1]
    
    # Remove 'c' if it is the last character and not part of a word
    if answer.endswith('.c') or answer.endswith(' c'):
        answer = answer[:-1]
    elif answer.endswith('c') and len(answer) > 1 and answer[-2] in ['.', ' ']:
        answer = answer[:-1]
    
    return answer.strip()

def answer_query(model, processor, video_path, query):
    container = av.open(video_path)
    total_frames = container.streams.video[0].frames
    indices = np.arange(0, total_frames, total_frames/8).astype(int)
    clip = read_video_pyav(container, indices)

    prompt = f"USER: <video>{query} ASSISTANT:"
    inputs = processor(text=prompt, videos=clip, return_tensors="pt")

    generate_ids = model.generate(
        **{k: v.to(model.device) for k, v in inputs.items()},
        max_new_tokens=90
    )

    full_answer = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]

    # Extract only the answer after "ASSISTANT:"
    answer = full_answer.split("ASSISTANT:")[1].strip()
    
    # Apply simplified cleaning
    answer = clean_answer(answer)

    return answer

def main(clips_dir, queries_json, output_json):
    with open(queries_json, 'r') as f:
        data = json.load(f)

    model, processor = load_model()
    results = []

    print(f"Starting processing of {len(data)} video clips...")
    
    # Progress bar for video processing
    for idx, item in enumerate(tqdm(data, desc="Processing", ncols=80)):
        uid = item["video_uid"]
        query = item["query"]
        pred_segment = item["pred"]
        clip_path = os.path.join(clips_dir, f"{uid}_clip_{idx:02d}.mp4")
        
        if not os.path.exists(clip_path):
            print(f"Clip not found: {clip_path}")
            continue

        try:
            answer = answer_query(model, processor, clip_path, query)
        except Exception as e:
            print(f"Error in clip {idx}: {e}")
            answer = ""

        results.append({
            "query": query,
            "video_uid": uid,
            "clip_path": clip_path,
            "pred": pred_segment,
            "answer": answer
        })

    print(f"Saving results to {output_json}...")
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("Processing completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--clips_dir", type=str, required=True, help="Directory containing video clips")
    parser.add_argument("--queries_json", type=str, required=True, help="JSON file with top50 queries")
    parser.add_argument("--output", type=str, required=True, help="Output JSON file for LLaVA answers")
    args = parser.parse_args()

    main(args.clips_dir, args.queries_json, args.output)

