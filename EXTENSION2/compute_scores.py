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
    bleu1_list, bleu4_list, rouge_list, meteor_list = [], [], [], []

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
        # BLEU-1
        bleu1_score = bleu.compute(predictions=[llava_answer], references=[[gt_answer]], max_order=1, smooth=True)["bleu"]
        # BLEU-4
        bleu4_score = bleu.compute(predictions=[llava_answer], references=[[gt_answer]], max_order=4, smooth=True)["bleu"]
        # ROUGE-L
        rouge_score = rouge.compute(predictions=[llava_answer], references=[[gt_answer]], use_stemmer=True)["rougeL"]
        # METEOR
        meteor_score = meteor.compute(predictions=[llava_answer], references=[[gt_answer]])["meteor"]

        # Store scores for averages
        bleu1_list.append(bleu1_score)
        bleu4_list.append(bleu4_score)
        rouge_list.append(rouge_score)
        meteor_list.append(meteor_score)

        # Append per-query results
        results.append({
            "query": query,
            "llava_answer": llava_answer,
            "gt_answer": gt_answer,
            "clip_path": clip_path,
            "bleu-1": round(bleu1_score, 4),
            "bleu-4": round(bleu4_score, 4),
            "rouge-L": round(rouge_score, 4),
            "meteor": round(meteor_score, 4)
        })

    # Compute averages
    avg_results = {
        "bleu-1": round(sum(bleu1_list)/len(bleu1_list), 4),
        "bleu-4": round(sum(bleu4_list)/len(bleu4_list), 4),
        "rouge-L": round(sum(rouge_list)/len(rouge_list), 4),
        "meteor": round(sum(meteor_list)/len(meteor_list), 4)
    }

    return results, avg_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--llava", type=str, required=True, help="JSON file with LLaVA answers")
    parser.add_argument("--gt", type=str, required=True, help="JSON file with ground-truth answers")
    args = parser.parse_args()

    results, avg_results = compute_scores(args.llava, args.gt)

    # Save per-query results
    with open("/content/score_results.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Print only average scores
    print("Average scores:")
    print(f"BLEU-1: {avg_results['bleu-1']}, BLEU-4: {avg_results['bleu-4']}, ROUGE-L: {avg_results['rouge-L']}, METEOR: {avg_results['meteor']}")

    print("Score results saved to /content/score_results.json")
