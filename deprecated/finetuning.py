import os
import yaml
import torch
from datasets import load_dataset, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, PeftModel
from training.preprocess import preprocess



desired_dir = "/llm-project/src/"
os.chdir(desired_dir)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(SCRIPT_DIR, "..", "config.yaml")
with open(CONFIG_PATH, 'r') as config_file:
    config = yaml.safe_load(config_file)
MODEL_NAME = config['finetuning']['model_name']
DATASET_PATH = os.path.join(SCRIPT_DIR, config['dataset']['dataset_training_path'])
TABLES_PATH = os.path.join(SCRIPT_DIR, config['dataset']['tables_training_path'])
TEST_QUERY = config['inference']['test_query']
MAX_TOKENS = config['model']['max_tokens']
EPOCHS = config['finetuning']['epochs']



def tokenize_function(examples, tokenizer):
    tokenized = tokenizer(
        examples["problem"],
        examples["answer"],
        truncation=True,
        padding="max_length",
        max_length=512
    )
    tokenized["labels"] = [ids[:] for ids in tokenized["input_ids"]]
    return tokenized



print(f"[1] Loading model...{MODEL_NAME}")
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)



print(f"[2] Adding padding tokens...")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id



if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template:
    print(f"[3] Using built-in chat template...")
else:
    print("[3] Defining custom chat template...")



print("[4] Configuring LoRA...")
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)



model = get_peft_model(model, lora_config)



print("[5] Loading and preparing dataset...")
dataset = preprocess(DATASET_PATH, TABLES_PATH, use_evidence=True)



print("[6] Tokenizing dataset...")
tokenized_datasets = dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True, remove_columns=dataset.column_names_original)



MODEL_FINETUNED_PATH = os.path.join(SCRIPT_DIR, MODEL_NAME + "_finetuned")
print("[7] Training arguments set.")
training_args = TrainingArguments(
    output_dir=MODEL_FINETUNED_PATH,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    num_train_epochs=EPOCHS,
    learning_rate=2e-4,
    fp16=True,
    save_steps=500,
    logging_steps=100,
    save_total_limit=2,
    report_to="none",
    label_names=["labels"],
)



print("[8] Fine tuning using LoRA...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets,
    processing_class=tokenizer,
    data_collator=DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    ),
)
trainer.train()



print("[9] Saving fine-tuned model...")
model.save_pretrained(MODEL_FINETUNED_PATH)
tokenizer.save_pretrained(MODEL_FINETUNED_PATH)



print("[10] Saving merged fine-tuned model with base model...")
MODEL_FINETUNED_MERGED_PATH = os.path.join(SCRIPT_DIR, MODEL_NAME + "_finetuned_merged")


model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = PeftModel.from_pretrained(model, MODEL_FINETUNED_PATH)
merged_model = model.merge_and_unload()
merged_model.save_pretrained(MODEL_FINETUNED_MERGED_PATH)
tokenizer.save_pretrained(MODEL_FINETUNED_MERGED_PATH)
print(f"Merged model saved to {MODEL_FINETUNED_MERGED_PATH}")