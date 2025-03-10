import numpy as np
from numpy.lib.stride_tricks import as_strided

class MaxPool2D:
    def __init__(self, pool_size, stride=None):
        self.pool_size = pool_size if isinstance(pool_size, tuple) else (pool_size, pool_size)
        self.stride = stride if stride is not None else self.pool_size
        self.X = None
        self.argmax = None

    def forward(self, X):
        """
        Forward pass for max pooling layer
        
        Args:
            X: Input of shape (N, C, H, W)
            
        Returns:
            out: Output of shape (N, C, H_out, W_out)
        """
        N, C, H, W = X.shape
        pool_h, pool_w = self.pool_size
        stride_h, stride_w = self.stride if isinstance(self.stride, tuple) else (self.stride, self.stride)
        
        # Calculate output dimensions
        out_h = (H - pool_h) // stride_h + 1
        out_w = (W - pool_w) // stride_w + 1
        
        # Store input for backward pass
        self.X = X
        
        # Reshape input to perform pooling
        X_reshaped = np.zeros((N, C, out_h, out_w, pool_h, pool_w))
        self.argmax = np.zeros((N, C, out_h, out_w, 2), dtype=int)
        
        # Fill in the reshaped tensor
        for i in range(out_h):
            for j in range(out_w):
                h_start = i * stride_h
                w_start = j * stride_w
                X_reshaped[:, :, i, j] = X[:, :, h_start:h_start+pool_h, w_start:w_start+pool_w]
        
        # Perform max pooling
        out = np.max(X_reshaped, axis=(4, 5))
        
        # Store argmax indices for backward pass
        for n in range(N):
            for c in range(C):
                for i in range(out_h):
                    for j in range(out_w):
                        h_start = i * stride_h
                        w_start = j * stride_w
                        window = X[n, c, h_start:h_start+pool_h, w_start:w_start+pool_w]
                        
                        # Get index of max value
                        idx = np.unravel_index(np.argmax(window), window.shape)
                        self.argmax[n, c, i, j] = idx
        
        return out

    def backward(self, dout):
        """
        Backward pass for max pooling layer
        
        Args:
            dout: Gradient from upstream of shape (N, C, H_out, W_out)
            
        Returns:
            dX: Gradient with respect to input X
        """
        N, C, H, W = self.X.shape
        pool_h, pool_w = self.pool_size
        stride_h, stride_w = self.stride if isinstance(self.stride, tuple) else (self.stride, self.stride)
        
        # Dimensions from forward pass
        _, _, out_h, out_w = dout.shape
        
        # Initialize gradient with respect to input
        dX = np.zeros_like(self.X)
        
        # Distribute gradients
        for n in range(N):
            for c in range(C):
                for i in range(out_h):
                    for j in range(out_w):
                        h_start = i * stride_h
                        w_start = j * stride_w
                        
                        # Get indices of max element
                        max_h, max_w = self.argmax[n, c, i, j]
                        
                        # Add gradient to the max element position
                        dX[n, c, h_start + max_h, w_start + max_w] += dout[n, c, i, j]
        
        return dX

class Dense:
    def __init__(self, input_size, output_size, activation = None):

        self.input_size = input_size
        self.output_size = output_size
        self.activation = activation
        scale = np.sqrt(2.0 / (input_size + output_size))
        self.weights = np.random.randn(input_size, output_size) * scale
        self.biases = np.zeros((1, output_size))
        self.input = None
        self.output = None

    def forward(self, X):
        self.input = X.reshape(X.shape[0], -1)  # Ensure this is stored
        self.output = self.input @ self.weights + self.biases
        if self.activation:
            self.output = self.activation.forward(self.output)
        return self.output
    
    def backward(self, dY):
        if self.activation:
            dY = self.activation.backward(dY)
        self.dW = self.input.T @ dY
        self.db = np.sum(dY, axis=0, keepdims=True)
        dX = dY @ self.weights.T
        # print(f"Dense dW max (inside backward): {np.max(np.abs(self.dW)):.4f}")  # Debug inside Dense
        return dX


class ReLU:
    
    def __init__(self):
        self.input = None       # Input for backward pass

    def forward(self, X):
        self.input = X
        return np.maximum(0, X)
    
    def backward(self, dY):
        return dY * (self.input > 0)        # ReLU derivative: 1 if input > 0, else 0
    

class Sigmoid:

    def __init__(self):
        self.output = None       # Output for use in backward pass

    def forward(self, X):
        self.output = 1 / (1 + np.exp(-X))
        return self.output
    
    def backward(self, dY):
        return dY * self.output * (1 - self.output)     # Sigmoid derivative: output * (1 - output)
                        

""" Convert input pathces to columns for vectorized convolution. """
def im2col_fixed(X, FH, FW, stride=1, padding=0):
    N, C, H, W = X.shape
    H_out = (H + 2 * padding - FH) // stride + 1
    W_out = (W + 2 * padding - FW) // stride + 1

    if padding > 0:
        X_padded = np.pad(X, ((0, 0), (0, 0), (padding, padding), (padding, padding)), mode='constant')
    else:
        X_padded = X

    cols = np.zeros((N * H_out * W_out, C * FH * FW))
    
    for i in range(H_out):
        for j in range(W_out):
            h_start = i * stride
            h_end = h_start + FH
            w_start = j * stride
            w_end = w_start + FW
            
            # Extract patches and reshape properly
            patch = X_padded[:, :, h_start:h_end, w_start:w_end]
            col = patch.reshape(N, -1)
            cols[i * W_out * N + j * N : i * W_out * N + (j + 1) * N, :] = col
            
    return cols, H_out, W_out

def col2im_fixed(cols, X_shape, FH, FW, stride=1, padding=0):
    N, C, H, W = X_shape
    H_out = (H + 2 * padding - FH) // stride + 1
    W_out = (W + 2 * padding - FW) // stride + 1
    
    X_padded = np.zeros((N, C, H + 2 * padding, W + 2 * padding))
    
    for i in range(H_out):
        for j in range(W_out):
            h_start = i * stride
            h_end = h_start + FH
            w_start = j * stride
            w_end = w_start + FW
            
            col = cols[i * W_out * N + j * N : i * W_out * N + (j + 1) * N, :]
            patch = col.reshape(N, C, FH, FW)
            X_padded[:, :, h_start:h_end, w_start:w_end] += patch
    
    if padding > 0:
        return X_padded[:, :, padding:-padding, padding:-padding]
    return X_padded

class Conv2D:
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding
        
        # He initialization
        FH, FW = self.kernel_size
        fan_in = in_channels * FH * FW
        scale = np.sqrt(2.0 / fan_in)
        self.weights = np.random.randn(out_channels, in_channels, FH, FW) * scale
        self.biases = np.zeros(out_channels)
        
        # Gradient storage
        self.dW = None
        self.db = None
        
        # Input cache for backward pass
        self.X_shape = None
        self.X_cols = None
        
    def forward(self, X):
        N, C, H, W = X.shape
        self.X_shape = X.shape
        
        FH, FW = self.kernel_size
        out_h = (H + 2 * self.padding - FH) // self.stride + 1
        out_w = (W + 2 * self.padding - FW) // self.stride + 1
        
        # Reshape weights for matrix multiplication
        W_reshaped = self.weights.reshape(self.out_channels, -1)
        
        # Convert input to columns
        X_cols, _, _ = im2col_fixed(X, FH, FW, self.stride, self.padding)
        self.X_cols = X_cols
        
        # Compute output
        out = W_reshaped @ X_cols.T + self.biases.reshape(-1, 1)
        out = out.reshape(self.out_channels, out_h, out_w, N)
        out = out.transpose(3, 0, 1, 2)
        
        return out
        
    def backward(self, dout):
        N, C, H, W = self.X_shape
        FH, FW = self.kernel_size
        
        # Reshape dout
        dout_reshaped = dout.transpose(1, 2, 3, 0).reshape(self.out_channels, -1)
        
        # Compute gradients
        self.dW = dout_reshaped @ self.X_cols
        self.dW = self.dW.reshape(self.weights.shape)
        self.db = np.sum(dout, axis=(0, 2, 3))
        
        # Backpropagate to input
        dX_cols = dout_reshaped.T @ self.weights.reshape(self.out_channels, -1)
        dX = col2im_fixed(dX_cols, self.X_shape, FH, FW, self.stride, self.padding)
        
        return dX
    
class LeakyReLU:
    def __init__(self, alpha=0.01):
        self.alpha = alpha
        self.X = None

    def forward(self, X):
        self.X = X
        return np.where(X > 0, X, self.alpha * X)

    def backward(self, dY):
        return np.where(self.X > 0, dY, self.alpha * dY)
