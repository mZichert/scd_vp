import sys
import os
import pandas as pd
import pickle
import numpy as np
from tqdm.notebook import tqdm
from itertools import chain
import datetime

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from accelerate import Accelerator
import math
from datasets import Dataset, load_from_disk

from transformers import (
    AutoTokenizer, #SciBERT 
    AutoModelForMaskedLM, #SciBERT
    AutoConfig, #SciBERT
    AutoTokenizer, #SciBERT
    BertConfig,
    BertForMaskedLM,
    BertTokenizer,
    BertModel,
    DataCollatorForLanguageModeling,
    default_data_collator,
    get_scheduler,
    Trainer,
    TrainingArguments
)

journals = ["pr", "pra", "prb", "prc", "prd", "pre", "prl", "rmp"]



# Choose and load model
def load_model(model_type, max_seq_length):
    if model_type == "bert":
        model = BertForMaskedLM.from_pretrained('bert-base-uncased', output_hidden_states = True)
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', use_fast=True, do_lower_case=True, max_len=max_seq_length)
    elif model_type == "scibert":
        model = AutoModelForMaskedLM.from_pretrained('allenai/scibert_scivocab_uncased', output_hidden_states = True)
        tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased', use_fast=True, do_lower_case=True, max_len=max_seq_length)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print("loaded model:", model.name_or_path)
    print("device:", model.device)
    model.eval()
    return model, tokenizer

# loads in all articles
# returns as list of sentences
def load_data(basepath, journals):
    sentences = []
    for journal in tqdm(journals):
        for path in tqdm(os.listdir(basepath + journal)):
            with open(basepath + journal + "/" + path, "r") as f:
                text = f.read()
            sentences.append(text.split("\n"))
    return sentences

def create_dataset(chunk_size, train_split, test_split):

    # Dataset erstellen

    dataset = Dataset.from_generator(generate_dataset_from_df)
    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

    # In Chunks der Größe chunksize aufteilen
    chunk_size = chunk_size
    chunked_dataset = tokenized_dataset.map(chunk_texts, batched=True)

    # Train und Testdaten festlegen
    train_size = int(len(chunked_dataset) * train_split)
    test_size = int(len(chunked_dataset) * test_split)
    downsampled_dataset = chunked_dataset.train_test_split(train_size=train_size, test_size=test_size)

    # Masking der gleichen Worte für Perplexity Vergleich
    if model_type == "scibert":
        downsampled_dataset = downsampled_dataset.remove_columns(["word_ids"])
    eval_dataset = downsampled_dataset["test"].map(insert_random_mask, batched=True, remove_columns=downsampled_dataset["test"].column_names)
    eval_dataset = eval_dataset.rename_columns({"masked_input_ids": "input_ids","masked_attention_mask": "attention_mask","masked_labels": "labels"})
    
    return downsampled_dataset, eval_dataset

# Generator for dataset-library
# for sentences > max_seq_length: only give first 512 tokens.
"""def generate_dataset():
    for journal in journals:
        for path in os.listdir(basepath + journal):
            with open(basepath + journal + "/" + path, "r") as f:
                text = f.read()    
            text = text.split("\n")
            for sent in text:
                if (len(sent.split(" ")) + sent.count(" ")) > 512:
                    continue
                else:
                    yield {"text" : sent.lower()}"""
                    
def generate_dataset_from_df():
    for text in df.text:
        paragraphs = text.split("\n\n")
        for p in paragraphs:
            for sent in p.split("\n"):
                if (len(sent.split(" ")) + sent.count(" ")) > 512:
                    continue
                else:
                    yield {"text" : sent}
            
# tokenizer for dataset-library
def tokenize_function(dataset):
    result = tokenizer(dataset["text"], max_length=chunk_size, truncation=True)
    if tokenizer.is_fast:
        result["word_ids"] = [result.word_ids(i) for i in range(len(result["input_ids"]))]
    return result

# create chunks with chunk_size
# Drop last chunk that is < chunk_size
def chunk_texts(examples):
    # Concatenate all texts
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    # Compute length of concatenated texts
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the last chunk if it's smaller than chunk_size
    total_length = (total_length // chunk_size) * chunk_size
    # Split by chunks of max_len
    result = {
        k: [t[i : i + chunk_size] for i in range(0, total_length, chunk_size)]
        for k, t in concatenated_examples.items()
    }
    # Create a new labels column
    result["labels"] = result["input_ids"].copy()
    return result

# Masking "statisch", das heißt vorher um Vergleichbarkeit zu garantieren
def insert_random_mask(batch):
    features = [dict(zip(batch, t)) for t in zip(*batch.values())]
    masked_inputs = data_collator(features)
    # Create a new "masked" column for each column in the dataset
    return {"masked_" + k: v.numpy() for k, v in masked_inputs.items()}


model_type = str(sys.argv[1])
fraction = float(sys.argv[2])

    
log_name = f"{model_type}_{fraction}_log"
with open(f"log/{log_name}.txt", "a") as f: f.write(f"Job started at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.\n\n")

# load model
max_seq_length = 512
#model_type = "scibert" #"scibert"
model, tokenizer = load_model(model_type, max_seq_length)
with open(f"log/{log_name}.txt", "a") as f: f.write(f"{model_type} loaded.\n\n")

# Data Collator plus Mask Probabilty
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)

### Create Dataset

df = pd.read_pickle("../../data/cleaned_texts/cleaned_texts_df.pkl")
journals = ["rmp", "pr", "pra", "prb", "prc", "prd", "pre", "prl"]
df = df[df.journal.isin(journals)]

if fraction != 1:
    # create random sample of texts
    df = df.sample(frac=fraction)

# create dataset
chunk_size = 128
train_dataset, eval_dataset = create_dataset(chunk_size=chunk_size, train_split=0.8, test_split=0.2)

if not os.path.exists(f'datasets/{model_type}/'):  
    os.makedirs(f'datasets/{model_type}/')
if not os.path.exists(f'datasets/{model_type}/{fraction}/'):  
    os.makedirs(f'datasets/{model_type}/{fraction}/') 
#train_dataset.save_to_disk(f'datasets/{model_type}/{fraction}/train')
#eval_dataset.save_to_disk(f'datasets/{model_type}/{fraction}/eval')

#with open(f"log/{log_name}.txt", "a") as f: f.write(f"Dataset created.\n")
#with open(f"log/{log_name}.txt", "a") as f: f.write(f"at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.\n\n")


# Training Arguments

batch_size = 32

training_args = TrainingArguments(
    output_dir=f"models/{model_type}-finetuned",
    num_train_epochs= 5,
    overwrite_output_dir= True,
    evaluation_strategy= "epoch",
    learning_rate= 5e-5, # for Adam Optimizer, wie bei kutuzov et al
    optim = "adamw_torch", # default = "adam_hf"
    adam_epsilon= 1e-8, # 1e-8 = Default, wie bei kutuzov et al
    weight_decay= 0.01, # kutuzov.et_al haben 0.00. Weight decay verhindert overfitting und scheint egl sinnvoll? https://towardsdatascience.com/weight-decay-and-its-peculiar-effects-66e0aee3e7b8
    per_device_train_batch_size = batch_size,
    per_device_eval_batch_size = batch_size,
    push_to_hub=False,
    fp16=True,  # Für Performance, 16-bit (mixed) precision training instead of 32 bit
    logging_strategy="epoch",
    save_strategy="epoch",
    #save_step = 500 # Default 500
    #logging_steps = logging_steps,
    report_to="none",
    lr_scheduler_type = "linear", # default linear wie bei kutuzov
    seed = 42,# default = 42
    auto_find_batch_size=False,
    load_best_model_at_end = True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset= train_dataset["train"],
    eval_dataset= eval_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer,
)

# Perplexity
eval_results = trainer.evaluate()
perplexity_before = round(math.exp(eval_results['eval_loss']), 2)
#print(f"{model_type} - perplexity before finetuning: {perplexity_before}.")
with open(f"log/{log_name}.txt", "a") as f: f.write(f"{model_type} - perplexity before finetuning: {perplexity_before}.\n")
#with open(f"log/{log_name}.txt", "a") as f: f.write(f"at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.\n\n")

# Train
trainer.train()
with open(f"log/{log_name}.txt", "a") as f: f.write(f"{model_type} - training finished.\n\n")

# Save best model
model.save_pretrained(f"models/{model_type}_{fraction}_finetuned_best/")

eval_results = trainer.evaluate()
perplexity_after = round(math.exp(eval_results['eval_loss']), 2)
#print(f"{model_type} - perplexity before finetuning: {perplexity_after}.")
with open(f"log/{log_name}.txt", "a") as f: f.write(f"{model_type} - perplexity after finetuning: {perplexity_after}.\n")

with open(f"log/{log_name}.txt", "a") as f: f.write(f"Job done at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.\n\n")