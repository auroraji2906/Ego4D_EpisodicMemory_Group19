# Ego4D_EpisodicMemory_Group19

This repository contains code for the **Natural Language Queries (NLQ)** task on egocentric videos from the Ego4D dataset, including VSLNet and VSLBase, with modified variants and an extension for qualitative evaluation using Video-LLaVA.

## Repository Structure

- `STEP4_TEMPORAL_LOCALIZATION/`  
  Notebooks for Step 4 of the project: training and evaluating VSLBase, VSLNet, and a variant of VSLNet with non-shared encoders for text and video, using EgoVLP or Omnivore features. The four notebooks included are:  
  - `VSLBase_EgoVLP.ipynb` – Train/evaluate VSLBase using EgoVLP features  
  - `VSLBase_Omnivore.ipynb` – Train/evaluate VSLBase using Omnivore features  
  - `VSLNet_EgoVLP_and_modified.ipynb` – Train/evaluate VSLNet and its variant using EgoVLP features  
  - `VSLNet_Omnivore_and_modified.ipynb` – Train/evaluate VSLNet and its variant using Omnivore features


- `STEP5_EXTENSION_LLaVA/`  
  Notebooks for Step 5 (qualitative extension):  
  - `Extension2.ipynb` – Execute Video-LLaVA on the top 50 selected queries and evaluate performance using BLEU-4, ROUGE-L, and METEOR scores.  
  - `Some_qualitative_results.ipynb` – Contains a subset of qualitative results, showing examples with both correct and incorrect Video-LLaVA answers compared to manually annotated ground truth.

- `NLQ/`  
  Contains all the necessary files to execute the notebooks.

## Notebooks

Following are all you need to successfully execute the notebooks:

### 1. `VSLBase_EgoVLP.ipynb` and `VSLBase_Omnivore.ipynb`
- **Dependencies:** all files in `NLQ/VSLBase/`  
- **Note:** make sure EgoVLP or Omnivore features are downloaded and stored in `/content/ego4d_data/` or you can download them from ego4d dataset using the secret key to access the dataset.

### 2. `VSLNet_Omnivore_and_modified.ipynb` and `VSLNet_EgoVLP_and_modified.ipynb`
- **Dependencies:** all files in `NLQ/VSLNet/`  
- **Execution:**  
  - We use `NLQ/VSLNet/main.py` for standard VSLNet  
  - We use `NLQ/VSLNet/main_VSLNet_modified.py` for the variant with separate encoders

    in the training cell (here we provide the code block for main.py)
    
    ```bash
    %%bash
    source vars.sh
    
    export DATALOADER_WORKERS=1
    export NUM_WORKERS=2
    export VAL_JSON_PATH="/content/ego4d_data/v1/annotations/nlq_val.json"
    
    export BATCH_SIZE=32
    export DIM=128
    export NUM_EPOCH=20
    export MAX_POS_LEN=128
    export INIT_LR=0.002
    
    export TB_LOG_NAME="${NAME}_bs${BATCH_SIZE}_dim${DIM}_epoch${NUM_EPOCH}_ilr${INIT_LR}"
    
    python main.py \
        --task $TASK_NAME \
        --predictor bert \
        --dim $DIM \
        --mode train \
        --video_feature_dim 256 \
        --max_pos_len $MAX_POS_LEN \
        --init_lr $INIT_LR \
        --epochs $NUM_EPOCH \
        --batch_size $BATCH_SIZE \
        --fv official \
        --num_workers $NUM_WORKERS \
        --data_loader_workers $DATALOADER_WORKERS \
        --model_dir $MODEL_BASE_DIR/$NAME \
        --eval_gt_json $VAL_JSON_PATH \
        --log_to_tensorboard $TB_LOG_NAME \
        --tb_log_freq 5 \
        --remove_empty_queries_from train
      ```
    
- **Note:** make sure EgoVLP or Omnivore features are downloaded and placed in `/content/ego4d_data/` or you can download them from ego4d dataset using the secret key to access the dataset.

### 3. `Some_qualitative_results.ipynb` and `Extension2.ipynb`
- **Dependencies:** all files in `NLQ/EXTENSION2/`  
  - `best_prediction.json`: best checkpoint from VSLNet on EgoVLP  
  - `compute_scores.py`: computes ROUGE, BLEU, and METEOR scores  
  - `extract_clips.py`: extracts video clips using ffmpeg  
  - `llava.py`: runs Video-LLaVA  
  - `select_query.py`: selects top 50 queries  
  - `top50_annotated.json`: manually annotated ground truth for top 50 queries (these are the answer that we manually annotate, or you can complete them by yourself watching the video temporal segment of that query). This file should contain a list of answers with this format:
 
    ```
    {
      "query": "What tool did I use on the machine first ",
      "query_idx": 0,
      "video_uid": "f681f510-cd33-48e3-bc10-4a8f2a518495",
      "clip_uid": "b8654118-84a4-4167-83c9-f268cc15f7b2",
      "annotation_uid": "633b4a69-98f3-4695-889c-28bda6a3a5fa",
      "pred": [
        0.0,
        26.25
      ],
      "iou": 0.9561378586882882,
      "answer": "The tool you used first on the machine was a screwdriver."
    },
    ```
    
    
- **Note:** for these notebooks you need to have access to the dataset to download the clips, so please make sure to have secret key, otherwise sign the Ego4D License at "[ego4ddataset.com](https://ego4ddataset.com)"

- **Execution:**
  
   `Extension2.ipynb`
  
     The following code cells (corresponds to substep 3,4 of extension) train Video-LLaVA and compute metric scores
  
    ```bash
    %%bash
    python /content/NLQ/EXTENSION2/llava.py \
        --clips_dir "/content/ego4d_data/v1/clips_top50" \
        --queries_json "/content/NLQ/EXTENSION2/top50_queries.json" \
        --output "/content/NLQ/EXTENSION2/answers_video_llava.json"
    
    %%bash
    python /content/NLQ/EXTENSION2/compute_scores.py \
        --llava "/content/NLQ/EXTENSION2/answers_video_llava.json" \
        --gt "/content/NLQ/EXTENSION2/top50_annotated.json"
    ```
  
  **Outputs:**
  
  - `answers_video_llava.json`: a list of answers generated by Video-LLaVA, that have the following format:
  
    ``` 
    {
      "query": "What tool did I use on the machine first ",
      "video_uid": "f681f510-cd33-48e3-bc10-4a8f2a518495",
      "clip_path": "/content/ego4d_data/v1/clips_top50/f681f510-cd33-48e3-bc10-4a8f2a518495_clip_01.mp4",
      "pred": [
        0.0,
        26.25
      ],
      "answer": "The tool you used first on the machine was a screwdriver."
    } ,
    ```
  
    where "video_uid" is the uid of the video that the query is related to, "clip_path" is the extracted clip from the video, using as [start,end] timestamp those provided by "pred", the temporal interval predicted by using the best checkpoint of VSLNet on EgoVLP.
  - `score_results.json`: contain a list of quantitative ROUGE, BLEU and METEOR scores for each query-answer that you need to analyze to choose a subset of interesting qualitative results.
  
    Example:
    ```
    {
      "query": "What tool did I use on the machine first",
      "llava_answer": "The tool you used first on the machine was a screwdriver.",
      "gt_answer": "The tool you used first on the machine was a screwdriver.",
      "clip_path": "/content/ego4d_data/v1/clips_top50/f681f510-cd33-48e3-bc10-4a8f2a518495_clip_01.mp4",
      "bleu": 1.0,
      "rouge-L": 1.0,
      "meteor": 0.9997
    },
    ```
  
  - `Some_qualitative_results.ipynb`: for each example, the notebook shows the query, the manually annotated answer, and the answer generated by Video-LLaVA. It also allows you to play the retrieved video clip corresponding to the query directly within the notebook using the `Video(...)` function.

**Important Notes**

- Ensure all files and folders from the original repositories are present as described.

- All notebooks require access to the downloaded features to run correctly.

- You must have access to the Ego4D dataset (with secret key) to download clips. Sign the Ego4D License at [Ego4D License](https://ego4ddataset.com/ego4d-license/) if needed.

- Recommended environment: Colab or a machine with GPU (T4/A100) for training and inference.
