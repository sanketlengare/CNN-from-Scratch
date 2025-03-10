from src.data import load_cifar10_train, load_cifar10_test, normalize_images
from src.model import CNN  # Rename this to your fixed CNN implementation
import numpy as np

def compute_accuracy(logits, labels):
    preds = np.argmax(logits, axis=1)
    return np.mean(preds == labels)

# Load and preprocess data - use a smaller subset for faster training
train_images, train_labels = load_cifar10_train()
test_images, test_labels = load_cifar10_test()

# Use only 10,000 training samples to reduce computation
train_size = 10000
train_images = train_images[:train_size]
train_labels = train_labels[:train_size]

# Use only 1,000 test samples
test_size = 1000
test_images = test_images[:test_size]
test_labels = test_labels[:test_size]

# Normalize
train_images = normalize_images(train_images)
test_images = normalize_images(test_images)
print(f"Train shape: {train_images.shape}, Test shape: {test_images.shape}")

# Split validation set - smaller validation set
val_size = 1000
train_images, val_images = train_images[:-val_size], train_images[-val_size:]
train_labels, val_labels = train_labels[:-val_size], train_labels[-val_size:]
print(f"Train subset shape: {train_images.shape}, Val shape: {val_images.shape}")

# Initialize the fixed CNN model with smaller architecture
cnn = CNN()  # Use your fixed model class

# Training hyperparameters
batch_size = 32  # Keep batch size smaller
learning_rate = 0.01
epochs = 5  # Fewer epochs
n_samples = train_images.shape[0]

# Simple training loop
for epoch in range(epochs):
    print(f"=== Epoch {epoch + 1}/{epochs} ===")
    
    # Shuffle the training data
    perm = np.random.permutation(n_samples)
    train_images_epoch = train_images[perm]
    train_labels_epoch = train_labels[perm]
    
    # Track metrics
    total_loss = 0
    batch_count = 0
    
    # Batch training
    for i in range(0, n_samples, batch_size):
        end_idx = min(i + batch_size, n_samples)
        batch_X = train_images_epoch[i:end_idx]
        batch_y = train_labels_epoch[i:end_idx]
        
        # Forward pass
        logits, loss = cnn.forward(batch_X, batch_y)
        
        # Track loss
        total_loss += loss
        batch_count += 1
        
        # Backward pass and update
        cnn.backward(batch_y)
        cnn.update_weights(learning_rate)
        
        # Print progress every 20 batches
        if (i // batch_size) % 20 == 0:
            acc = compute_accuracy(logits, batch_y)
            print(f"Batch {i//batch_size}: Loss = {loss:.4f}, Acc = {acc:.4f}")
            
            # Debug info to check if loss is decreasing
            if hasattr(cnn, 'loss_fn') and hasattr(cnn.loss_fn, 'probs'):
                probs = cnn.loss_fn.probs
                print(f"  Probability stats - max: {np.max(probs):.4f}, min: {np.min(probs):.4f}")
                print(f"  Logits range: {np.min(logits):.4f} to {np.max(logits):.4f}")
    
    # Calculate average training loss
    avg_train_loss = total_loss / batch_count
    
    # Evaluate on validation set
    val_logits, val_loss = cnn.forward(val_images, val_labels)
    val_acc = compute_accuracy(val_logits, val_labels)
    
    print(f"Epoch {epoch + 1} summary:")
    print(f"  Avg train loss: {avg_train_loss:.4f}")
    print(f"  Val loss: {val_loss:.4f}, Val accuracy: {val_acc:.4f}")
    
    # Check if loss is decreasing (not plateauing at 2.3)
    print(f"  Is model learning? {'Yes' if val_loss < 2.29 else 'No'}")

# Final evaluation on test set
test_logits, test_loss = cnn.forward(test_images, test_labels)
test_acc = compute_accuracy(test_logits, test_labels)

print("\nFinal Test Results:")
print(f"Test loss: {test_loss:.4f}, Test accuracy: {test_acc:.4f}")

# Print class distribution of predictions to check for collapsed predictions
test_preds = np.argmax(test_logits, axis=1)
unique, counts = np.unique(test_preds, return_counts=True)
print("\nPrediction distribution:")
for class_idx, count in zip(unique, counts):
    print(f"Class {class_idx}: {count} predictions ({count/len(test_preds)*100:.1f}%)")