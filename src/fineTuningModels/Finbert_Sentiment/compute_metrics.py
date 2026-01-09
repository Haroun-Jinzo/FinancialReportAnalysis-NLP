from typing import List
import numpy as np
import json
import logging
from pathlib import Path
from datetime import datetime

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score
)

logger = logging.getLogger(__name__)

def compute_metrics(eval_pred):

    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return {
        'accuracy': accuracy_score(labels, predictions),
        'f1_macro': f1_score(labels, predictions, average='macro'),
        'f1_weighted': f1_score(labels, predictions, average='weighted'),
        'precision': precision_score(labels, predictions, average='weighted', zero_division=0),
        'recall': recall_score(labels, predictions, average='weighted', zero_division=0),
    }


class LoggingCallback:

    def __init__(self, log_file: str):
        self.log_file = log_file
        self.training_history = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            self.training_history.append({
                'step': state.global_step,
                'epoch': state.epoch,
                **logs
            })
            
            # Log to file
            with open(self.log_file, 'a') as f:
                f.write(f"{datetime.now()}: {json.dumps(logs)}\n")
    
    def on_train_end(self, args, state, control, **kwargs):
        # Save training history
        history_file = Path(self.log_file).parent / 'training_history.json'
        with open(history_file, 'w') as f:
            json.dump(self.training_history, f, indent=2)
    
