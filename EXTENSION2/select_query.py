''' Selects the top-k queries based on the Intersection over Union (IoU) between predicted and ground-truth temporal segments and save them in a JSON file. '''

import json
import os

def compute_IoU(pred_segment, gt_segment):
    start1, end1 = pred_segment
    start2, end2 = gt_segment
    intersection = max(0, min(end1, end2) - max(start1, start2))
    union = max(end1, end2) - min(start1, start2)
    return intersection / union if union > 0 else 0

def select_topk_queries(pred_file_path, val_file_path, output_path, k=50):
    # Load predictions
    with open(pred_file_path, 'r') as f:
        preds_list = json.load(f)['results']

    # Index predictions: (clip_uid, annotation_uid, query_idx) → prediction
    pred_map = {}
    for p in preds_list:
        key = (p['clip_uid'], p.get('annotation_uid'), p['query_idx'])
        if 'predicted_times' in p and p['predicted_times']:
            pred_map[key] = p['predicted_times'][0]

    # Load ground truth
    with open(val_file_path, 'r') as f:
        val_data = json.load(f)['videos']

    candidates = []

    for video in val_data:
        video_uid = video['video_uid']
        for clip in video['clips']:
            clip_uid = clip['clip_uid']
            #clip_start = clip['clip_start_sec']
            #clip_end = clip['clip_end_sec']

            for annotation in clip['annotations']:
                annotation_uid = annotation['annotation_uid']
                for query_idx, query in enumerate(annotation['language_queries']):
                    if 'video_start_sec' not in query or 'video_end_sec' not in query:
                        continue

                    key = (clip_uid, annotation_uid, query_idx)
                    if key not in pred_map:
                        continue

                    pred_segment = pred_map[key]
                    gt_segment = [query['video_start_sec'], query['video_end_sec']]
                    iou_value = compute_IoU(pred_segment, gt_segment)

                    candidates.append({
                        'query': query['query'],
                        'query_idx': query_idx,
                        'video_uid': video_uid,
                        'clip_uid': clip_uid,
                        'annotation_uid': annotation_uid,
                        'pred': pred_segment,
                        'iou': iou_value,
                        #'clip_start_sec': clip_start,
                        #'clip_end_sec': clip_end,
                        'answer': ''
                    })

    # Sort by descending IoU
    candidates.sort(key=lambda x: x['iou'], reverse=True)

    # Take top k (default 50)
    selected = candidates[:k]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(selected, f, indent=2)

    print(f"Saved top {k} candidate queries to {output_path}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_file", type=str, required=True, help="Path to prediction JSON file")
    parser.add_argument("--val_file", type=str, required=True, help="Path to validation JSON file")
    parser.add_argument("--output", type=str, required=True, help="Output path to save selected queries")
    parser.add_argument("--k", type=int, default=50, help="Number of candidates to save for manual annotation")
    args = parser.parse_args()

    select_topk_queries(args.pred_file, args.val_file, args.output, args.k)

