import pickle
from tqdm import tqdm
import pandas as pd
from collections import Counter, defaultdict
from itertools import chain, groupby, combinations
import os
import numpy as np
import gc

import sys

from sklearn import metrics
from sklearn.cluster import AffinityPropagation, KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cosine
from scipy import stats

from sklearn.metrics import silhouette_score



def create_slice_df(data_df, year, slice_len, journal):
    if journal:
        return data_df.loc[(data_df.year >= year) & (data_df.year < year+slice_len) & (data_df.journal == journal)]
    else:
        return data_df.loc[(data_df.year >= year) & (data_df.year < year+slice_len)]

### Von calable and Interpretable Semantic Change Detection, Montariol2021
# https://github.com/matejMartinc/scalable_semantic_shift/blob/master/measure_semantic_shift.py
def aff_prop(slice_df, column, preference):
    embeddings = np.array(slice_df[column])
    embeddings = [np.array(emb.tolist()) for emb in embeddings]
    AP = AffinityPropagation(
            preference=preference,
            damping=0.9, #default 0.5, bei 0.75 gibt es keinen Convergence Error
            max_iter = 400, #default 200
            convergence_iter = 50, #default 15
            affinity = "euclidean", # ‘euclidean’, ‘precomputed’ defaul ‘euclidean’
            random_state=None)
    AP.fit(embeddings)
    #cluster_centers_indices = AP.cluster_centers_indices_
    cluster_centers = AP.cluster_centers_
    labels = AP.labels_
    return labels, cluster_centers

### Von calable and Interpretable Semantic Change Detection, Montariol2021
# https://github.com/matejMartinc/scalable_semantic_shift/blob/master/measure_semantic_shift.py
def aff_prop(slice_df, column, preference):
    embeddings = np.array(slice_df[column])
    embeddings = [np.array(emb.tolist()) for emb in embeddings]
    AP = AffinityPropagation(
            preference=preference,
            damping=0.9, #default 0.5, bei 0.75 gibt es keinen Convergence Error
            max_iter = 400, #default 200
            convergence_iter = 50, #default 15
            affinity = "euclidean", # ‘euclidean’, ‘precomputed’ defaul ‘euclidean’
            random_state=None)
    AP.fit(embeddings)
    #cluster_centers_indices = AP.cluster_centers_indices_
    cluster_centers = AP.cluster_centers_
    labels = AP.labels_
    return labels, cluster_centers

# von montariol2021 (leicht verändert)
def combine_clusters(labels, embeddings, threshold, remove):
    cluster_embeds = defaultdict(list)
    for label, embed in zip(labels, embeddings):
        cluster_embeds[label].append(embed)
    min_num_examples = threshold
    legit_clusters = []
    for idd, num_examples in Counter(labels).items():
        if num_examples >= threshold:
            legit_clusters.append(idd)
        if idd not in remove and num_examples < min_num_examples:
            min_num_examples = num_examples
            min_cluster_id = idd

    if len(set(labels)) == 2:
        return labels
    
    min_dist = 1
    all_dist = []
    cluster_labels = ()
    embed_list = list(cluster_embeds.items())
    for i in range(len(embed_list)):
        for j in range(i+1,len(embed_list)):
            idd, embed = embed_list[i]
            id2, embed2 = embed_list[j]
            if idd in legit_clusters and id2 in legit_clusters:
                dist = compute_averaged_embedding_dist(embed, embed2)
                all_dist.append(dist)
                if dist < min_dist:
                    min_dist = dist
                    cluster_labels = (idd, id2)
    std = np.std(all_dist)
    avg = np.mean(all_dist)
    limit = avg - 4 * std
    #limit = avg - 2 * std
    #limit = 0.1
    if min_dist < limit:
        for n, i in enumerate(labels):
            if i == cluster_labels[0]:
                labels[n] = cluster_labels[1]
        return combine_clusters(labels, embeddings, threshold, remove)

    if min_num_examples >= threshold:
        return labels

    min_dist = 1
    cluster_labels = ()
    for idd, embed in cluster_embeds.items():
        if idd != min_cluster_id:
            dist = compute_averaged_embedding_dist(embed, cluster_embeds[min_cluster_id])
            if dist < min_dist:
                min_dist = dist
                cluster_labels = (idd, min_cluster_id)

    if cluster_labels[0] not in legit_clusters:
        for n, i in enumerate(labels):
            if i == cluster_labels[0]:
                labels[n] = cluster_labels[1]
    else:
        if min_dist < limit:
            for n, i in enumerate(labels):
                if i == cluster_labels[0]:
                    labels[n] = cluster_labels[1]
        else:
            remove.append(min_cluster_id)
    return combine_clusters(labels, embeddings, threshold, remove)

def compute_averaged_embedding_dist(t1_embeddings, t2_embeddings):
    t1_mean = np.mean(t1_embeddings, axis=0)
    t2_mean = np.mean(t2_embeddings, axis=0)
    dist = cosine(t1_mean, t2_mean)
    return dist


def k_means(slice_df, column, k, random_state):
    embeddings = np.array(slice_df[column])
    embeddings = [np.array(emb.tolist()) for emb in embeddings]
    KM = KMeans(n_clusters=k, 
                init = "k-means++", #default = "k-means++"
                n_init = "auto",
                random_state=random_state).fit(embeddings)
    labels = KM.labels_
    cluster_centers = KM.cluster_centers_
    return labels, cluster_centers

#journals = ["pr", "pra", "prb", "prc", "prd", "pre", "prl", "rmp"]


    

token_df = pd.read_pickle("../../data/embeddings/virtual_fulltexts_virtual_token_embeddings.pkl")

token_type = str(sys.argv[1])
cluster_method = str(sys.argv[2])
journal = str(sys.argv[3])
if journal == "all": 
    journal = None
    token_df = token_df.sample(frac=0.66)


if journal:
    token_df = token_df.loc[token_df.journal == journal]
    
# remove duplicate sentences
if token_type == "sentence_emb":
    token_df["check_duplicates"] = token_df.sentence.apply(lambda x: " ".join(x))
    token_df = token_df.drop_duplicates(subset=["check_duplicates", "doi"]).drop("check_duplicates", axis=1)
if token_type == "token_emb":
    token_df = token_df.loc[token_df.token=="virtual"]

if cluster_method == "ap":
    
    token_df["ap_cluster"] = None
    token_df["ap_cluster_filtered"] = None
    
    # Affinity Propagation
    ap_labels, cluster_centers = aff_prop(token_df, token_type, preference=None)
    for target_id, label in zip(token_df.index, ap_labels):
        token_df.at[target_id, "ap_cluster"] = int(label)
    # filter and combine clusters
    ap_f_labels = combine_clusters(ap_labels, token_df[token_type], threshold=10, remove=[])
    for target_id, label in zip(token_df.index, ap_f_labels):
        token_df.at[target_id, "ap_cluster_filtered"] = int(label)
        
if cluster_method == "km":
    
    k = 10
    
    token_df["k_means"] = None
    token_df["k_means_filtered"] = None

    # K-Means clustering
    km_labels, cluster_centers = k_means(token_df, token_type, k=k, random_state=None)
    for target_id, label in zip(token_df.index, km_labels):
        token_df.at[target_id, f"k_means"] = int(label)
    # filter and combine clusters
    km_f_labels = combine_clusters(np.array(km_labels.tolist()), token_df[token_type], threshold=10, remove=[])
    for target_id, label in zip(token_df.index, km_f_labels):
        token_df.at[target_id, f"k_means_filtered"] = int(label)
    

if token_type == "token_emb":
    token_df.drop("sentence_emb", axis=1).to_pickle(f"../04_analysis/clustering/all_years/{token_type}_{cluster_method}_clustering_{journal if journal else 'all'}.pkl")
if token_type == "sentence_emb":
    token_df.drop("token_emb", axis=1).to_pickle(f"../04_analysis/clustering/all_years/{token_type}_{cluster_method}_clustering_{journal if journal else 'all'}.pkl")