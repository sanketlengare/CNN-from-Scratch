import numpy as np


class CrossEntropyLoss:
    def __init__(self):
        self.probs = None
        self.labels = None
        self.batch_size = None
        
    def forward(self, logits, labels):
        self.batch_size = logits.shape[0]
        self.labels = labels
        
        # For numerical stability
        logits_shifted = logits - np.max(logits, axis=1, keepdims=True)
        exp_logits = np.exp(logits_shifted)
        self.probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        
        # Cross-entropy loss
        correct_log_probs = -np.log(self.probs[np.arange(self.batch_size), labels])
        loss = np.mean(correct_log_probs)
        
        return loss
        
    def backward(self):
        # Gradient of softmax with cross-entropy
        d_logits = self.probs.copy()
        d_logits[np.arange(self.batch_size), self.labels] -= 1
        d_logits /= self.batch_size
        
        return d_logits