import os
import json
import torch
import requests

import pandas as pd
import numpy as np

from tqdm import tqdm
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from sklearn.metrics import accuracy_score, mean_absolute_error

################################################

DIMENSIONS = ['Noise', 'Sharpness', 'Details', 'Dynamic Range', 'Exposure', 
              'Contrast', 'Banding', 'White Balance', 'Saturation', 'Ghost']

SCORE_TO_LABEL = {
    -2: "Motorola < Monalisa",
    -1: "Motorola =< Monalisa",
    0:  "Motorola = Monalisa",
    1:  "Motorola => Monalisa",
    2:  "Motorola > Monalisa"
}

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

model_id = "Qwen/Qwen2-VL-7B-Instruct"
processor = AutoProcessor.from_pretrained(model_id)
model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_id, quantization_config=bnb_config, torch_dtype=torch.bfloat16, device_map="cuda"
)

################################################

def text_to_scores(text_description):
    # Construct the prompt for the "Collator" task
    prompt = f"""<|system|>
    You are a precision data extraction assistant. Your task is to compare the quality of Image A and Image B based on a text description.

    SCORING RULES:
    -2: Image A is much worse than Image B
    -1: Image A is slightly worse than Image B
    0: Image A and Image B are equal
    1: Image A is slightly better than Image B
    2: Image A is much better than Image B

    REQUIRED OUTPUT FORMAT:
    Return ONLY a valid JSON object with these exact keys: "Noise", "Sharpness", "Details", "Dynamic Range", "Exposure", "Contrast", "Banding", "White Balance", "Saturation", "Ghost".

    DESCRIPTION:
    {text_description}
    <|user|>
    Extract the scores."""

    # Qwen2-VL chat template format
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    inputs = processor(text=[text], return_tensors="pt").to("cuda")

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=128)
        # Trim the input tokens from the output
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)]
        response_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    try:
        # Strip potential markdown backticks
        clean_text = response_text.strip()
        if clean_text.startswith("```json"):
            clean_text = clean_text.split("```json")[1].split("```")[0].strip()
        elif clean_text.startswith("```"):
            clean_text = clean_text.split("```")[1].strip()
            
        return json.loads(clean_text)
    except Exception as e:
        # Debugging: See why it failed
        print(f"Failed to parse for description: {text_description[:50]}... Error: {e}")
        return {}

def label_to_score(val):

    """Parses ground truth string label into a numerical score."""
    
    if pd.isna(val): return np.nan
    val = str(val).strip()
    if '=>' in val or '>=' in val: return 1
    if '=<' in val or '<=' in val: return -1
    if '<' in val: return -2
    if '>' in val: return 2
    if '=' in val: return 0
    return np.nan

def generate_predictions_csv(json_path, output_csv_path):

    """
    Takes model output JSON, calls the LLM  for each pair and saves the results to a CSV file.
    """

    with open(json_path, 'r') as f:
        data = json.load(f)
        
    predictions_list = []
    print(f"Generating predictions for {len(data['image_pairs'])} pairs... ( :")

    for pair in tqdm(data['image_pairs']):
        scene_id = str(pair['scene_id']).zfill(3)
        raw_text = pair['text_description']
        
        # json LLM call
        scores = text_to_scores(raw_text) 
        
        row_data = {'scene_id': scene_id}
        for dim in DIMENSIONS:
            
            score = scores.get(dim.lower(), scores.get(dim, 0)) 
            row_data[dim] = SCORE_TO_LABEL.get(score, "Motorola = Monalisa") # tie for nothing
            
        predictions_list.append(row_data)
        
    df_preds = pd.DataFrame(predictions_list)
    df_preds.to_csv(output_csv_path, index=False)
    print(f"Created predictions file: {output_csv_path}")


#############################
# run the thing

for ind, example in enumerate(os.listdir('inputs')):

    length = len(os.listdir('inputs'))
    print(f'Doing {ind+1}/{length}...\n')

    example = example.replace('.json', '')
    input_path = f'inputs/{example}.json'
    output_path = f'results/{example}.csv'

    generate_predictions_csv(input_path, output_path)
