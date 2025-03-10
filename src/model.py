import numpy as np
from src.layers import Conv2D, MaxPool2D, Dense, LeakyReLU
from src.loss import CrossEntropyLoss  # Import from loss.py

class Dropout:
    def __init__(self, p=0.5):
        self.p = p
        self.mask = None

    def forward(self, X, training=True):
        if training:
            self.mask = np.random.binomial(1, 1 - self.p, X.shape) / (1 - self.p)
            return X * self.mask
        return X

    def backward(self, dY):
        return dY * self.mask

class CNN:
    def __init__(self):
        # Use proper padding to maintain spatial dimensions
        self.conv1 = Conv2D(3, 32, 3, 1, 1)  # Output: 32x32
        self.relu1 = LeakyReLU()
        self.conv2 = Conv2D(32, 64, 3, 1, 1)  # Output: 32x32
        self.relu2 = LeakyReLU()
        self.pool1 = MaxPool2D((2, 2), 2)  # Output: 16x16
        
        # Proper size calculation: 64 filters * 16*16 feature map size
        self.dense1 = Dense(64 * 16 * 16, 10)
        self.loss_fn = CrossEntropyLoss()
        
        self.trainable_layers = [self.conv1, self.conv2, self.dense1]
        self.velocity = {layer: {'W': np.zeros_like(layer.weights), 
                                'b': np.zeros_like(layer.biases)} 
                        for layer in self.trainable_layers}
        
    def forward(self, X, labels=None):
        X = self.conv1.forward(X)
        X = self.relu1.forward(X)
        X = self.conv2.forward(X)
        X = self.relu2.forward(X)
        X = self.pool1.forward(X)
        
        X = X.reshape(X.shape[0], -1)  # Flatten for dense layer
        logits = self.dense1.forward(X)
        
        if labels is not None:
            loss = self.loss_fn.forward(logits, labels)
            return logits, loss
        return logits
        
    def backward(self, labels):
        dX = self.loss_fn.backward()
        dX = self.dense1.backward(dX)
        dX = dX.reshape(-1, 64, 16, 16)  # Reshape to match pool1 output
        dX = self.pool1.backward(dX)
        dX = self.relu2.backward(dX)
        dX = self.conv2.backward(dX)
        dX = self.relu1.backward(dX)
        dX = self.conv1.backward(dX)
        return dX
        
    def update_weights(self, learning_rate=0.01, momentum=0.9, weight_decay=0.0001):
        for layer in self.trainable_layers:
            # Apply weight decay
            dW = layer.dW + weight_decay * layer.weights
            
            # Apply momentum update (no gradient clipping)
            self.velocity[layer]['W'] = momentum * self.velocity[layer]['W'] - learning_rate * dW
            layer.weights += self.velocity[layer]['W']
            
            self.velocity[layer]['b'] = momentum * self.velocity[layer]['b'] - learning_rate * layer.db
            layer.biases += self.velocity[layer]['b']
