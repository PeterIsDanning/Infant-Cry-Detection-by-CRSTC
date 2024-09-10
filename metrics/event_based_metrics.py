import sklearn
import numpy as np
import matplotlib.pyplot as plt

# Args:
# y_true (ndarray): Ground truth labels (0 or 1).
# y_pred (ndarray): Predicted labels (0 or 1).

# Event-based metrics
def event_metrics(y_true, y_pred, tolerance, overlap_threshold=0.7, switch=False):
    if switch:
        y_pred = (1 - y_pred > 0)
    else:
        y_pred = y_pred > 0
    # Create empty list for storing true events
    true_events = []
    # Initilize start index
    start = None
    for i, label in enumerate(y_true):
        if label == 1 and start is None:
            start = i
        elif label == 0 and start is not None:
            true_events.append((start, i - 1))
            start = None
    
    if start is not None:
        true_events.append((start, len(y_true) - 1))

    pred_events = []
    start = None
    for i, label in enumerate(y_pred):
        if label == 1 and start is None:
            start = i
        elif label == 0 and start is not None:
            pred_events.append((start, i - 1))
            start = None

    if start is not None:
        pred_events.append((start, len(pred_events) - 1))
        

    # Highlight overlapping events
    # Intialize true positive and overlap events
    tp, fp, fn = 0, 0, 0
    counted_events = []
    fake_events = []
    undetected_events = []
    pred_check = pred_events[:]

    iou_list = []
    for true_event in true_events:
        tp_event = 0
        for pred_event in pred_events:
            lower_bound = true_event[0] - tolerance
            upper_bound = true_event[1] + tolerance
            # Calculate overlap rate
            overlap_rate = 0
            if lower_bound <= pred_event[0] and upper_bound >= pred_event[1]:
                overlap_start = max(true_event[0], pred_event[0])
                overlap_end = min(true_event[1], pred_event[1])
                overlap_length = overlap_end - overlap_start + 1
                true_length = true_event[1] - true_event[0] + 1
                pred_length = pred_event[1] - pred_event[0] + 1
                overlap_rate = overlap_length / min(true_length, pred_length)

            # Range check
            if overlap_rate >= overlap_threshold:
                union_start = min(true_event[0], pred_event[0])
                union_end = max(true_event[1], pred_event[1])
                union_length = union_end - union_start + 1
                iou = overlap_length / union_length
                iou_list.append(iou)
                # True positive: correctly detected events
                if pred_event in pred_check:
                    pred_check.remove(pred_event)
                    if tp_event == 0:
                        tp_event = 1
                        counted_events.append((true_event[0], true_event[1]))

        # False negative: events in true label that have not been correctly detected according to the definition
        if tp_event == 0:
            fn += 1
            undetected_events.append((true_event[0], true_event[1]))

        tp += tp_event
    # False positive: events in prediction that are not correct according to the definition
    if pred_check:
        for pred_event in pred_check: 
            if pred_event[1] - pred_event[0] > tolerance:
                fp += 1
                fake_events.append((pred_event[0], pred_event[1]))

    if tp == 0 and fn == 0 and fp == 0:
        F = 1
    else:
        # Calculation of F-Score
        P = tp / (tp + fp) if (tp + fp) != 0 else 0
        R = tp / (tp + fn) if (tp + fn) != 0 else 0
        F = 2 * P * R / (P + R) if (P + R) != 0 else 0
    # Calculation of IOU
    if iou_list == []:
        IOU = 0
    else:    
        IOU = np.mean(iou_list)
    return F, IOU, counted_events, fake_events, undetected_events

def event_visualization(y_true, y_pred, counted_events, fake_events, undetected_events):
    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(y_true)), y_true, label='True Label')
    plt.plot(range(len(y_pred)), y_pred, label='Predicted Label')

    for event in counted_events:
        plt.axvspan(event[0], event[1], alpha=0.3, color='green', label='Overlap event')
    for event in fake_events:
        plt.axvspan(event[0], event[1], alpha=0.3, color='red', label='Fake event')
    for event in undetected_events:
        plt.axvspan(event[0], event[1], alpha=0.3, color='blue', label='Undetected event')

    # Add labels and title
    plt.xlabel('Index')
    plt.ylabel('Label')
    plt.title('Overlapping Events Visualization')
    plt.legend()
    plt.grid(True)
    plt.show()

# Reference: https://doi.org/10.3390/app6060162