import numpy as np
import librosa
import torch
import matplotlib.pyplot as plt

from metrics.event_based_metrics import event_metrics
from src.audio_preprocessing import readLabels, object_padding, fbank_features_extraction
from tslearn.clustering import TimeSeriesKMeans, KShape
from tslearn.metrics import dtw
from sklearn.cluster import KMeans, AffinityPropagation, AgglomerativeClustering, MeanShift, estimate_bandwidth, DBSCAN, OPTICS, Birch
from sklearn.metrics import accuracy_score, f1_score

import os
os.environ["OMP_NUM_THREADS"] = '3'

def standardize_array(array):
    mean = np.mean(array, axis=0)
    std = np.std(array, axis=0)

    # Avoid division by zero
    std[std == 0] = 1
    standardized_array = (array - mean) / std
    return standardized_array

# Functions to cluster and label a single audio's frames
def kmeans_clustering(audio_data, n_clusters=2):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(audio_data)
    labels = kmeans.predict(audio_data)
    return labels

def dtw_kmedoids_clustering(audio_data, n_clusters=2):
    km = TimeSeriesKMeans(n_clusters=n_clusters, metric="dtw", max_iter=10, random_state=42)
    labels = km.fit_predict(audio_data)
    return labels

def kshape_clustering(audio_data, n_clusters=2):
    ks = KShape(n_clusters=n_clusters, max_iter=10, random_state=42)
    labels = ks.fit_predict(audio_data)
    return labels

def affinity_propagation_clustering(audio_data):
    af = AffinityPropagation(random_state=42)  
    labels = af.fit_predict(audio_data)
    return labels

def agglomerative_clustering(audio_data, n_clusters=2):
    agg = AgglomerativeClustering(n_clusters=n_clusters, metric='precomputed', linkage='average') 
    distances = [[dtw(x, y) for y in audio_data] for x in audio_data]
    labels = agg.fit_predict(distances)
    return labels

def mean_shift_clustering(audio_data):
    bandwidth = estimate_bandwidth(audio_data, quantile=0.2) 
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True, cluster_all=False)
    labels = ms.fit_predict(audio_data)
    return labels

def bisecting_kmeans_clustering(audio_data, n_clusters=2):
    clusters = [audio_data]
    while len(clusters) < n_clusters:
        largest_cluster_idx = max(range(len(clusters)), key=lambda i: len(clusters[i]))
        largest_cluster = clusters[largest_cluster_idx]

        km = TimeSeriesKMeans(n_clusters=2, metric="dtw", max_iter=10, random_state=42)
        sub_labels = km.fit_predict(largest_cluster)
        sub_cluster1 = largest_cluster[sub_labels == 0]
        sub_cluster2 = largest_cluster[sub_labels == 1]

        clusters.pop(largest_cluster_idx)
        clusters.append(sub_cluster1)
        clusters.append(sub_cluster2)

    labels = [-1] * len(audio_data) 
    for i, cluster in enumerate(clusters):
        for idx in [j for j, x in enumerate(audio_data) if x in cluster]:
            labels[idx] = i

    return np.array(labels)

def dbscan_clustering(audio_data, eps=0.5, min_samples=5):
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric=dtw)
    labels = dbscan.fit_predict(audio_data)
    return labels

def optics_clustering(audio_data, min_samples=5):
    optics = OPTICS(min_samples=min_samples, metric=dtw, cluster_method='xi')
    labels = optics.fit_predict(audio_data)
    return labels

def birch_clustering(audio_data, n_clusters=None, branching_factor=50, threshold=0.5):
    birch = Birch(n_clusters=n_clusters, branching_factor=branching_factor, threshold=threshold)
    labels = birch.fit_predict(audio_data)
    return labels

def clustering_predicting(model, annotation_file, audio_file, max_length, clustering_method="kmeans", k=2):
    signal, fs = librosa.load(audio_file)
    signal = object_padding(signal, max_length)
    truth_labels = readLabels(path=annotation_file, sample_rate=fs)
    truth_labels = object_padding(truth_labels, max_length)
     
    test_audio = fbank_features_extraction([audio_file], max_length)

    test_input = torch.tensor(test_audio, dtype=torch.float32)
    x_recon, test_latent, test_u, loss = model(test_input)

    clustering_input = standardize_array(test_u.reshape((703, -1)).detach().numpy())
    clustering_label = None

    if clustering_method == "kmeans":
        clustering_label = kmeans_clustering(clustering_input, n_clusters=k)
    elif clustering_method == "dtw":
        clustering_label = dtw_kmedoids_clustering(clustering_input, n_clusters=k)
    elif clustering_method == "kshape":
        clustering_label = kshape_clustering(clustering_input, n_clusters=k)
    elif clustering_method == "affinity":
        clustering_label = affinity_propagation_clustering(clustering_input)
    elif clustering_method == "agglomerative":
        clustering_label = agglomerative_clustering(clustering_input, n_clusters=k)
    elif clustering_method == "mean_shift":
        clustering_label = mean_shift_clustering(clustering_input)
    elif clustering_method == "bisecting":
        clustering_label = bisecting_kmeans_clustering(clustering_input, n_clusters=k)
    elif clustering_method == "DBSCAN":
        clustering_label = dbscan_clustering(clustering_input)
    elif clustering_method == "OPTICS":
        clustering_label = optics_clustering(clustering_input)
    elif clustering_method == "Birch":
        clustering_label = birch_clustering(clustering_input, n_clusters=k)

    label_timeseries = np.zeros(max_length)
    begin = int(0)
    end = int(0.025 *fs)
    shift_step = int(0.01 * fs)
    for i in range(clustering_label.shape[0]):
        label_timeseries[begin:end] = abs(clustering_label[i])
        begin = begin + shift_step
        end = end + shift_step

    return signal, fs, np.array(truth_labels), label_timeseries

def signal_visualization(signal, fs, truth_labels, label_timeseries):
    # define time axis
    Ns = len(signal) # number of sample
    Ts = 1 / fs # sampling period
    t = np.arange(Ns) * Ts # time axis in seconds
    norm_coef = 1.1 * np.max(signal)
    edge_ind = np.min([signal.shape[0], len(truth_labels)])
     
    plt.figure(figsize=(24, 6))
    plt.plot(t[:edge_ind], signal[:edge_ind])
    plt.plot(t[:edge_ind], truth_labels[:edge_ind] * norm_coef)
    plt.plot(t[:edge_ind], label_timeseries[:edge_ind] * norm_coef)

    plt.title("Ground truth labels")
    plt.legend(['Signal', 'Cry', 'Clusters'])
    plt.show()

def cluster_visualization(signal, fs, truth_labels, label_timeseries):
    # define time axis
    Ns = len(signal)  # number of sample
    Ts = 1 / fs  # sampling period
    t = np.arange(Ns) * Ts  # time axis in seconds
    norm_coef = 1.1 * np.max(signal)
    edge_ind = np.min([signal.shape[0], len(truth_labels)])

    plt.figure(figsize=(24, 6))
    line_signal, = plt.plot(t[:edge_ind], signal[:edge_ind])

    # Identify 'cry' and 'non-cry' segments
    cry_indices = np.where(truth_labels == 1)[0]
    non_cry_indices = np.where(truth_labels == 0)[0]

    # Fill rectangular segments for 'cry' and 'non-cry'
    # Identify start and end points for each continuous segment
    start_cry = np.insert(np.where(np.diff(cry_indices) != 1)[0] + 1, 0, 0)
    end_cry = np.append(np.where(np.diff(cry_indices) != 1)[0], len(cry_indices) - 1)

    # Fill rectangular segments
    for start, end in zip(start_cry, end_cry):
        plt.fill_between(
            t[cry_indices[start:end+1]], 
            0, 
            norm_coef, 
            color='orange', 
            alpha=0.5,  # Adjust transparency as needed
            label='Cry' if start == start_cry[0] else None  # Avoid duplicate labels
        )
    legend_handles = []
    legend_handles.append(plt.Rectangle((0, 0), 1, 1, color='orange', alpha=0.5))

    start_non_cry = np.insert(np.where(np.diff(non_cry_indices) != 1)[0] + 1, 0, 0)
    end_non_cry = np.append(np.where(np.diff(non_cry_indices) != 1)[0], len(non_cry_indices) - 1)   

    # Fill rectangular segments
    for start, end in zip(start_non_cry, end_non_cry):
        plt.fill_between(
            t[non_cry_indices[start:end+1]], 
            0, 
            norm_coef, 
            color='gray', 
            alpha=0.5,  # Adjust transparency as needed
            label='Non-cry' if start == start_non_cry[0] else None  # Avoid duplicate labels
        )
    legend_handles.append(plt.Rectangle((0, 0), 1, 1, color='gray', alpha=0.5))

    # Get unique values in label_timeseries to assign distinct colors
    unique_labels = np.unique(label_timeseries)
    cmap = plt.get_cmap('tab10')  # You can choose other colormaps as needed

    # Fill rectangular segments for each unique label
    for i, label in enumerate(unique_labels):
        label_indices = np.where(label_timeseries == label)[0] 
        
        # Identify start and end points for each continuous segment
        start_indices = np.insert(np.where(np.diff(label_indices) != 1)[0] + 1, 0, 0)
        end_indices = np.append(np.where(np.diff(label_indices) != 1)[0], len(label_indices) - 1)

        # Fill rectangular segments
        for start, end in zip(start_indices, end_indices):
            plt.fill_between(
                t[label_indices[start:end+1]], 
                0, 
                -norm_coef, 
                color=cmap(i), 
                alpha=0.5,  # Adjust transparency as needed
                label=f'Cluster {label}' if start == start_indices[0] else None  # Avoid duplicate labels
            )
        legend_handles.append(plt.Rectangle((0, 0), 1, 1, color=cmap(i), alpha=0.5))

    plt.title("Audio Clustering")
    plt.legend(
        [line_signal] + legend_handles, 
        ['Signal'] + ['Cry', 'Non-Cry'] + [f'Cluster {label}' for label in unique_labels]
    )
    plt.show()

def clustering_evaluatation(model, max_length, audio_files, annotation_files, domain_index=None, clustering_method="kmeans", k=2):
    acc_list, framef_list, eventf_list, iou_list = [], [], [], []
    switch_list = []
    if domain_index is None:
        domain_index = range(len(audio_files))
    for i in domain_index:
        annotation_file = annotation_files[i]
        audio_file = audio_files[i]
        clustering_switch = False
        _, _, truth_labels, label_timeseries = clustering_predicting(model, annotation_file, audio_file, max_length, clustering_method, k)
        temp_accuracy = accuracy_score(truth_labels, label_timeseries)
        framef = max(f1_score(1 - label_timeseries > 0, truth_labels), f1_score(label_timeseries > 0, truth_labels))
        if temp_accuracy < 0.5:
            clustering_accuracy = 1-temp_accuracy
            clustering_switch = True
        else:
            clustering_accuracy = temp_accuracy
        acc_list.append(clustering_accuracy)
        switch_list.append(clustering_switch)
        framef_list.append(framef)
        eventf, iou, _, _, _ = event_metrics(truth_labels, label_timeseries, tolerance=2000, overlap_threshold=0.75)
        eventf_list.append(eventf)
        iou_list.append(iou)

    return acc_list, framef_list, eventf_list, iou_list, switch_list