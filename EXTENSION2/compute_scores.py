'''Computes BLEU, ROUGE and METEOR scores for LLaVA answers against ground-truth references and stores the evaluation results in a JSON file.'''
import json
import evaluate
import argparse

def compute_scores(llava_path, gt_path):
    with open(llava_path, "r") as f:
        llava_results = json.load(f)
    with open(gt_path, "r") as f:
        gt_data = json.load(f)

    llava_dict = {item["query"].strip(): item for item in llava_results}

    # Load evaluation metrics
    bleu = evaluate.load("bleu")
    rouge = evaluate.load("rouge")
    meteor = evaluate.load("meteor")

    results = []
    for gt_item in gt_data:
        query = gt_item["query"].strip()
        gt_answer = gt_item["answer"].strip()

        if query not in llava_dict:
            print(f"Warning: query '{query}' not found in LLaVA results.")
            continue

        llava_item = llava_dict[query]
        llava_answer = llava_item["answer"].strip()
        clip_path = llava_item.get("clip_path", "")

        # Compute evaluation metrics
        bleu_score = bleu.compute(predictions=[llava_answer], references=[[gt_answer]])["bleu"]
        rouge_score = rouge.compute(predictions=[llava_answer], references=[[gt_answer]])["rougeL"]
        meteor_score = meteor.compute(predictions=[llava_answer], references=[[gt_answer]])["meteor"]
        
        # Create the results
        results.append({
            "query": query,
            "llava_answer": llava_answer,
            "gt_answer": gt_answer,
            "clip_path": clip_path,
            "bleu": bleu_score,
            "rouge-L": rouge_score,
            "meteor": meteor_score
        })
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--llava", type=str, required=True, help="JSON file with LLaVA answers")
    parser.add_argument("--gt", type=str, required=True, help="JSON file with ground-truth answers")
    args = parser.parse_args()

    results = compute_scores(args.llava, args.gt)

    with open("/content/score_results.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("Score results saved to /content/score_results.json")
