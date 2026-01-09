import numpy as np
from seqeval.metrics import precision_score, recall_score, f1_score

# ==================== METRICS ====================

def compute_ner_metrics(predictions, labels, id2label):
    """
    Compute NER-specific metrics using seqeval
    
    seqeval properly handles entity boundaries
    """
    # Convert IDs to labels
    pred_labels = []
    true_labels = []
    
    for pred_seq, true_seq in zip(predictions, labels):
        pred_seq_labels = []
        true_seq_labels = []
        
        for pred_id, true_id in zip(pred_seq, true_seq):
            if true_id != -100:  # Ignore padding
                pred_seq_labels.append(id2label[pred_id])
                true_seq_labels.append(id2label[true_id])
        
        pred_labels.append(pred_seq_labels)
        true_labels.append(true_seq_labels)
    
    # Compute metrics
    return {
        'precision': precision_score(true_labels, pred_labels),
        'recall': recall_score(true_labels, pred_labels),
        'f1': f1_score(true_labels, pred_labels)
    }


def compute_metrics_wrapper(id2label):
    """Wrapper to pass id2label to compute_metrics"""
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=2)
        
        return compute_ner_metrics(predictions, labels, id2label)
    
    return compute_metrics