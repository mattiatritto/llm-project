import os
import json
import torch
from datasets import load_dataset, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from preprocess import preprocess
import yaml
from test_time_scaling import decode_output

desired_dir = "/llm-project/src/"
os.chdir(desired_dir)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(SCRIPT_DIR, "..", "config.yaml")
with open(CONFIG_PATH, 'r') as config_file:
    config = yaml.safe_load(config_file)
MODEL_NAME = config['inference']['model_name']
DATASET_PATH = os.path.join(SCRIPT_DIR, config['dataset']['dataset_dev_path'])
TABLES_PATH = os.path.join(SCRIPT_DIR, config['dataset']['tables_dev_path'])
TEST_QUERY = config['inference']['test_query']
MAX_TOKENS = config['model']['max_tokens']
NUM_QUERIES = config['inference']['num_queries']
DEVICE = config['inference']['device']
DECODING_STRATEGY = config['inference']['decoding_strategy']



def predict(dataset, model, use_evidence, tokenizer, num_queries=NUM_QUERIES):
    print(f"[3] Tokenizing dataset (evidence = {use_evidence})...")
    tokenized_datasets = dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)

    if num_queries == "ALL":
        num_queries = len(tokenized_datasets)
    else:
        num_queries = min(num_queries, len(tokenized_datasets))
    
    print(f"[4] Running inference on the first {num_queries} rows of the dataset (evidence = {use_evidence}, decoding = {DECODING_STRATEGY})...")
    queries = tokenized_datasets.select(range(num_queries))

    results = {}
    for i, query in enumerate(queries):
        problem = query["problem"]
        db_id = query["db_id"]

        if i % 10 == 0:
            print("[Progress status]: Executed ", i, " queries")
        
        inputs = tokenizer(problem, return_tensors="pt").to("cuda")
        generated_sql = decode_output(model, inputs, db_id, tokenizer, strategy=DECODING_STRATEGY)
        generated_sql = generated_sql.replace("\n", " ")
        formatted_output = f"{generated_sql}\t----- bird -----\t{db_id}"
        results[str(i)] = formatted_output

    print(f"[5] Saving results to predict.json for eval (evidence = {use_evidence})...")

    EVAL_EVIDENCE_PATH = os.path.join(SCRIPT_DIR, "..", "evaluation", "results", "evidence", "predict.json")
    EVAL_NO_EVIDENCE_PATH = os.path.join(SCRIPT_DIR, "..", "evaluation", "results", "no_evidence", "predict.json")

    if use_evidence == True:
        with open(EVAL_EVIDENCE_PATH, "w") as f:
            json.dump(results, f, indent=4)
    else:
        with open(EVAL_NO_EVIDENCE_PATH, "w") as f:
            json.dump(results, f, indent=4)



def tokenize_function(examples, tokenizer):
    return tokenizer(examples["problem"], examples["answer"], truncation=True, padding="max_length", max_length=512)



torch.device(DEVICE)
print(f"[0] Using device: {torch.cuda.get_device_name(0)}")



print(f"[1] Loading model...{MODEL_NAME}")
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)



print("[2] Loading and preparing dataset...")
dataset_with_evidence = preprocess(DATASET_PATH, TABLES_PATH, use_evidence=True)
dataset_without_evidence = preprocess(DATASET_PATH, TABLES_PATH, use_evidence=False)



predict(dataset_with_evidence, model, use_evidence=True, tokenizer=tokenizer)
predict(dataset_without_evidence, model, use_evidence=False, tokenizer=tokenizer)