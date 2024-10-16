import pandas as pd
from itertools import chain
import os
from collections import Counter
import pickle
import numpy as np
import datetime

import torch
from transformers import BertTokenizer, BertModel 
from transformers import AutoTokenizer, AutoModel # for SciBERT

from transformers import logging
logging.set_verbosity_warning()

from nltk.corpus import stopwords

import sys


# load and choose model
def load_model(path, model_type):
    if model_type == "bert":
        model = BertModel.from_pretrained(path, output_hidden_states = True)
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    elif model_type == "scibert":
        model = AutoModel.from_pretrained(path, output_hidden_states = True)
        tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print("loaded model:", model.name_or_path)
    print("device:", model.device)
    model.eval()
    return model, tokenizer

# tokenize text
def get_tokenized_text(df, doi, tokenizer):
    sents = []
    for paragraph in df.loc[doi].text.split("\n\n"): 
        for sent in paragraph.split("\n"):
            sents.append(sent.lower())
    # add special tokens
    sents = ["[CLS] " + sent + " [SEP]" for sent in sents]
    # tokenize using input tokenizer
    tokenized_sents = [tokenizer.tokenize(sent) for sent in sents]
    return tokenized_sents

# bekommt tokenized_sents und erstellt batches mit gleichlangen sequenzen für jeden paragraph
# Tokenizing passiert auch hier
def tokens_to_batches(tokenized_texts, batch_size, chunk_size):

    # slice text into sequence with chunk_size
    # Einheiten sind Sätze. Wenn es nicht aufgeht wird der Rest mit PAD gefüllt
    batch = []
    
    for tokenized_sents in tokenized_texts:
        
        sequence = []
        
        for i, sent in enumerate(tokenized_sents):
            # Fülle auf solange Sequenz < chunk_size
            if len(sent) + len(sequence) <= chunk_size:
                for token in sent:
                    sequence.append(token)
                # Für letzten Satz
                if i == len(tokenized_sents)-1:
                    for pad in range(chunk_size - len(sequence)):
                        sequence.append("[PAD]")
                    batch.append((sequence, # tokenized text
                              tokenizer.convert_tokens_to_ids(sequence))) # token ids
            # Wenn Satz > chunk_size, fülle bis chunk_size und starte neue Sequenze
            elif len(sent) >= chunk_size:
                for token in sent:
                    if len(sequence) < chunk_size:
                        sequence.append(token)
                    if len(sequence) == chunk_size:
                        batch.append((sequence, # tokenized text
                              tokenizer.convert_tokens_to_ids(sequence))) # token ids
                        sequence = []
            # Fülle Rest mit [PAD], wenn es Satzende zulässt
            else:
                for pad in range(chunk_size - len(sequence)):
                    sequence.append("[PAD]")
                batch.append((sequence, # tokenized text
                              tokenizer.convert_tokens_to_ids(sequence))) # token ids
                sequence = []
                for token in sent:
                    sequence.append(token)

        """# catch last sequence
        if len(batch) == 0 or batch[-1][0] != sequence:
            for pad in range(chunk_size - len(sequence)):
                sequence.append("[PAD]")
            batch.append((sequence, # tokenized text
                          tokenizer.convert_tokens_to_ids(sequence))) # token ids"""
        
    ## create batches
    batches = []
    for i in range(0, len(batch), batch_size):
        batches.append(batch[i:i+batch_size])

    return batches   

# Use model to get embeddings
# Summe der letzten 4 Layers des Encoder Outputs für jeden Token
def get_token_embeddings(batches, model, chunk_size):
    
        
    token_embeddings = []
    tokenized_text = []

    for batch in batches:
        batch_size = len(batch)
        if torch.cuda.is_available():
            tokens_tensor = torch.zeros(batch_size, chunk_size, dtype=torch.long).cuda()
            segments_tensors = torch.ones(batch_size, chunk_size, dtype=torch.long).cuda()
        else:
            tokens_tensor = torch.zeros(batch_size, chunk_size, dtype=torch.long).cpu()
            segments_tensors = torch.ones(batch_size, chunk_size, dtype=torch.long).cpu()

        batch_idx = [x[1] for x in batch]
        batch_tokens = [x[0] for x in batch]

        for i in range(batch_size):
            for j in range(chunk_size):
                tokens_tensor[i][j] = batch_idx[i][j]

        # Predict hidden states features for each layer
        with torch.no_grad():
            model_output = model(tokens_tensor, token_type_ids=segments_tensors)
            encoded_layers = model_output[-1][-4:] #last four layers of the encoder

        # Sum last 4 Layers for each token
        for batch_i in range(batch_size):

            for token_i in range(chunk_size):

                # skip [PAD]
                if batch_tokens[batch_i][token_i] == "[PAD]":
                    continue

                # Holds last 4 layers of hidden states for each token
                hidden_layers = []
                for layer_i in range(len(encoded_layers)):
                    # Lookup the vector for `token_i` in `layer_i`
                    vec = encoded_layers[layer_i][batch_i][token_i]
                    hidden_layers.append(vec)

                # Sum last 4 layers 
                hidden_layers_sum = torch.sum(torch.stack(hidden_layers)[-4:], 0).reshape(1, -1).detach().cpu().numpy()

                token_embeddings.append(hidden_layers_sum)
                tokenized_text.append(batch_tokens[batch_i][token_i])
                    
    return tokenized_text, token_embeddings

# Subword-Tokens zu Wordvektoren aggregieren, dh Durchschnitt nehmen
def make_wordvectors(tokens, embeddings):
    
    # Subwordtokens zusammenziehen und embedding averagen
    new_txt = []
    new_emb = []
    prev_token = False
    temp_array = np.zeros((1, 768))
    temp_text = []
    i = 0
    for token, emb in zip(tokens, embeddings):

        if token.startswith("##"):
            if not prev_token:
                temp_text.append(new_txt[i-1])
                temp_text.append(token)
                temp_array += new_emb[i-1]
                temp_array += emb
                prev_token = True
            else:
                temp_text.append(token)
                temp_array += emb
        else:
            if prev_token:
                new_emb[i-1] = temp_array / len(temp_text)
                temp_array = np.zeros((1, 768))
                new_txt[i-1] = "".join(temp_text).replace("##", "")
                temp_text = []
            new_txt.append(token)
            new_emb.append(emb)
            i+=1
            prev_token = False

    # Tokens, die mit "-" getrennt sind, zusammenziehen und embeddings averagen
    iplus1 = -1
    iplus2 = -1
    result_tokens = []
    result_embs = []
    temp_array = np.zeros((1, 768))
    for i, (token, emb) in enumerate(zip(new_txt, new_emb)):
        if i == iplus1:
            iplus1 = -1
            continue
        if i == iplus2:
            iplus2 = -1
            continue
        if i+2 < len(new_txt):
            if new_txt[i+1] == "-":
                result_tokens.append("".join([token, new_txt[i+1], new_txt[i+2]]))
                temp_array += emb
                temp_array += new_emb[i+1]
                temp_array += new_emb[i+2]
                result_embs.append(temp_array / 3)
                temp_array = np.zeros((1, 768))
                iplus1 = i + 1
                iplus2 = i + 2
                continue
        if token == "-":
            if i - 1 >= 0 and i + 1 <  len(new_txt):
                result_tokens[-1] = result_tokens[-1] + "-" + new_txt[i+1]
                temp_array += emb
                temp_array += new_emb[i+1]
                temp_array = temp_array / 2
                result_embs[-1] = (result_embs[-1] + temp_array) / 2
                temp_array = np.zeros((1, 768))
                iplus1 = i + 1
                continue
        result_tokens.append(token)
        result_embs.append(emb)
        
    return result_tokens, result_embs


# filter out some tokens
# reduce embeddings to float16
def filter_and_reduce_wordvectors(tokens, embeddings, stop_words):
    
    return_tokens = []
    return_embeddings = []
    
    for token, emb in zip(tokens, embeddings):
        
        if token not in stop_words and len(token) > 3 and not token.isdecimal():
            return_tokens.append(token)
            #return_embeddings.append(np.float16(emb[0]))
            return_embeddings.append(np.float32(emb[0]))
            #return_embeddings.append(emb[0])
        
    return return_tokens, return_embeddings

# Durchschnitt der Embeddings nehmen pro Token
def make_average(tokens, embeddings):
    
    data_dict = {token : [] for token in tokens}
    
    for token, embedding in zip(tokens, embeddings):
        data_dict[token].append(embedding)
        
    return_tokens = []
    return_embeddings = []
    for token, embs in data_dict.items():
        temp_array = np.zeros((1, 768))
        for array in embs:
            temp_array += array
        return_tokens.append(token)
        return_embeddings.append((temp_array / len(embs))[0])
    
    return return_tokens, return_embeddings

year = int(sys.argv[1])

with open(f"__log.txt", "a") as f: 
    f.write(f"{year} started at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.\n")

df = pd.read_pickle("../../data/cleaned_texts/cleaned_texts_df.pkl")

df = df.loc[df.year == year].copy()

# Load best finetuned model by perplexity score
model_type = "bert"
path = "models/bert_1_finetuned_best/"

model, tokenizer = load_model(path, model_type)

batch_size = 16
chunk_size = 512

stop_words = set(stopwords.words("english"))
stop_words.update({"[SEP]", ",", ".", ":", ";"})

for year, year_slice in df.groupby("year"):
    
    # tokenize text
    tokenized_texts = []
    for doi in year_slice.index:
        tokenized_sents = get_tokenized_text(df, doi, tokenizer)
        tokenized_texts.append(tokenized_sents)
        
    # Create tokenized batches for training
    batches = tokens_to_batches(tokenized_texts, batch_size, chunk_size)
    # Train and get word vectors
    tokens, embeddings = get_token_embeddings(batches, model, chunk_size)
     # Wordvektoren zusammenziehen
    tokens, embeddings = make_wordvectors(tokens, embeddings)
    # Filter tokens and reduce size of embeddings to float32
    tokens, embeddings = filter_and_reduce_wordvectors(tokens, embeddings, stop_words)
    # Get average embeddings for every token
    tokens, embeddings = make_average(tokens, embeddings)
    # save to file
    save_dict = {token : emb for token,emb in zip(tokens, embeddings)}
    with open(f"../../data/embeddings/years_average/{year}.pkl", "wb") as f:
        pickle.dump(save_dict, f)
        
    with open(f"__log.txt", "a") as f: f.write(f"{year} done at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.\n")