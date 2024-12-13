import os
import time
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import warnings
import torch.nn.functional as F

# Local imports
from data_loader import HSIDataLoader
from spectralformer import SpectralFormer
from loss import HSIAdvLoss

# Configuration
CONFIG = {
    'data': {
        'dataset': 'Salinas',
        'data_dir': 'Data',
        'patch_size': 7,
        'batch_size': 64,
        'train_size' : .2,
        'val_size' : .2
    },
    'training': {
        'num_epochs': 80,
        'learning_rate': 0.0001,
        'weight_decay': 0.0001,
        'patience': 30
    },
    'paths': {
        'checkpoints': 'checkpoints',
        'visualizations': 'Visualizations',
        'results': 'Results'
    },
    'files': {
        'training_plot': 'Visualizations/training_history.png',
        'confusion_matrix': 'Results/confusion_matrix.png',
        'best_model': 'checkpoints/best_model.pth'
    }
}

def setup_directories():
    """Create necessary directories for outputs."""
    for path in CONFIG['paths'].values():
        # Create each directory if it doesn't exist
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
            print(f"Created directory: {path}")

def plot_training_history(history, save_path):
    """Plot training history with adversarial metrics."""
    plt.figure(figsize=(20, 5))
    
    # Loss plot
    plt.subplot(141)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.plot(history['train_adv_loss'], label='Train Adv Loss', linestyle='--')
    plt.plot(history['val_adv_loss'], label='Val Adv Loss', linestyle='--')
    plt.title('Losses over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Accuracy plot
    plt.subplot(142)
    plt.plot(history['train_acc'], label='Train Acc')
    plt.plot(history['val_acc'], label='Val Acc')
    plt.plot(history['train_adv_acc'], label='Train Adv Acc', linestyle='--')
    plt.plot(history['val_adv_acc'], label='Val Adv Acc', linestyle='--')
    plt.title('Accuracy over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    # Adversarial ratio plot
    plt.subplot(143)
    train_adv_ratio = [adv/std if std != 0 else 0 
                      for adv, std in zip(history['train_adv_loss'], history['train_loss'])]
    val_adv_ratio = [adv/std if std != 0 else 0 
                     for adv, std in zip(history['val_adv_loss'], history['val_loss'])]
    plt.plot(train_adv_ratio, label='Train Adv/Std Ratio')
    plt.plot(val_adv_ratio, label='Val Adv/Std Ratio')
    plt.title('Adversarial/Standard Loss Ratio')
    plt.xlabel('Epoch')
    plt.ylabel('Ratio')
    plt.legend()
    plt.grid(True)
    
    # Time plot
    plt.subplot(144)
    plt.plot(history['epoch_times'], label='Epoch Time')
    plt.title('Training Time per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Time (seconds)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Save plot
    plt.savefig(save_path)
    plt.close()
    print(f"Saved training history plot to: {save_path}")

def plot_confusion_matrix(conf_matrix, class_names, save_path):
    """Plot and save confusion matrix."""
    plt.figure(figsize=(12, 10))
    sns.heatmap(confusion_matrix, annot=True, fmt='.0f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch with center pixel classification."""
    metrics_sum = {
        'standard_loss': 0.0,
        'accuracy': 0.0,
        'adversarial_loss': 0.0,
        'adversarial_accuracy': 0.0,
        'total_loss': 0.0
    }
    num_batches = len(train_loader)
    
    train_progress = tqdm(train_loader, desc='Training')
    for batch_idx, (inputs, labels) in enumerate(train_progress):
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        loss, batch_metrics = criterion(inputs, labels, training=True)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        for key in metrics_sum:
            if key in batch_metrics:
                metrics_sum[key] += batch_metrics[key]
        
        train_progress.set_postfix_str(criterion.get_metrics_string(batch_metrics))
    
    return {k: v / num_batches for k, v in metrics_sum.items()}

def validate_epoch(model, val_loader, criterion, device):
    """Validate for one epoch with center pixel predictions."""
    metrics_sum = {
        'standard_loss': 0.0,
        'accuracy': 0.0,
        'adversarial_loss': 0.0,
        'adversarial_accuracy': 0.0,
        'total_loss': 0.0
    }
    num_batches = len(val_loader)
    
    model.eval()
    val_progress = tqdm(val_loader, desc='Validation')
    
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(val_progress):
            inputs, labels = inputs.to(device), labels.to(device)
            
            loss, batch_metrics = criterion(inputs, labels, training=True)
            
            for key in metrics_sum:
                if key in batch_metrics:
                    metrics_sum[key] += batch_metrics[key]
            
            val_progress.set_postfix_str(criterion.get_metrics_string(batch_metrics))
    
    return {k: v / num_batches for k, v in metrics_sum.items()}

def update_history(history, train_metrics, val_metrics, epoch_time):
    """
    Update training history with latest metrics.
    
    Args:
        history: Dictionary containing training history
        train_metrics: Training metrics for current epoch
        val_metrics: Validation metrics for current epoch
        epoch_time: Time taken for the epoch
    """
    history['train_loss'].append(train_metrics['standard_loss'])
    history['train_acc'].append(train_metrics['accuracy'])
    history['train_adv_loss'].append(train_metrics.get('adversarial_loss', 0))
    history['train_adv_acc'].append(train_metrics.get('adversarial_accuracy', 0))
    history['val_loss'].append(val_metrics['standard_loss'])
    history['val_acc'].append(val_metrics['accuracy'])
    history['val_adv_loss'].append(val_metrics.get('adversarial_loss', 0))
    history['val_adv_acc'].append(val_metrics.get('adversarial_accuracy', 0))
    history['epoch_times'].append(epoch_time)

def save_checkpoint(model, checkpoint_path):
    """
    Save model checkpoint with just the model state dict.
    """
    torch.save(model.state_dict(), checkpoint_path)
    print(f'Saved model state to {checkpoint_path}')

def print_epoch_summary(epoch, num_epochs, train_metrics, val_metrics, epoch_time):
    """Print summary of epoch results with all metrics."""
    print(f'\nEpoch {epoch+1}/{num_epochs}:')
    
    print('Training Metrics:')
    print(f"Standard Loss: {train_metrics['standard_loss']:.4f}")
    print(f"Accuracy: {train_metrics['accuracy']:.4f}")
    print(f"Adversarial Loss: {train_metrics['adversarial_loss']:.4f}")
    print(f"Adversarial Accuracy: {train_metrics['adversarial_accuracy']:.4f}")
    print(f"Total Loss: {train_metrics['total_loss']:.4f}")
    
    print('\nValidation Metrics:')
    print(f"Standard Loss: {val_metrics['standard_loss']:.4f}")
    print(f"Accuracy: {val_metrics['accuracy']:.4f}")
    print(f"Adversarial Loss: {val_metrics['adversarial_loss']:.4f}")
    print(f"Adversarial Accuracy: {val_metrics['adversarial_accuracy']:.4f}")
    print(f"Total Loss: {val_metrics['total_loss']:.4f}")
    
    print(f'\nEpoch Time: {epoch_time:.2f}s')

def train_model(model, train_loader, val_loader, device, config):
    """Train the model and return training history."""
    # Initialize tracking
    best_val_loss = float('inf')
    patience_counter = 0
    history = {
        'train_loss': [], 'train_acc': [], 'train_adv_loss': [], 'train_adv_acc': [],
        'val_loss': [], 'val_acc': [], 'val_adv_loss': [], 'val_adv_acc': [],
        'epoch_times': []
    }
    
    # Initialize training components
    criterion = HSIAdvLoss(model, epsilon=0.01, alpha=0.0)
    optimizer = optim.AdamW(model.parameters(), 
                           lr=config['training']['learning_rate'],
                           weight_decay=config['training']['weight_decay'])
    scheduler = CosineAnnealingLR(optimizer, 
                                 T_max=config['training']['num_epochs'],
                                 eta_min=1e-6)
    
    print("\nStarting training...")
    for epoch in range(config['training']['num_epochs']):
        epoch_start = time.time()
        
        # Training phase
        model.train()
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validation phase
        model.eval()
        val_metrics = validate_epoch(model, val_loader, criterion, device)
        
        # Update learning rate
        scheduler.step()
        
        # Update history
        update_history(history, train_metrics, val_metrics, time.time() - epoch_start)
        
        # Save best model
        if val_metrics['total_loss'] < best_val_loss:
            best_val_loss = val_metrics['total_loss']
            save_checkpoint(model, os.path.join(config['paths']['checkpoints'], 'best_model.pth'))
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Print epoch summary
        print_epoch_summary(epoch, config['training']['num_epochs'], 
                          train_metrics, val_metrics, time.time() - epoch_start)
        
        # Early stopping check
        if patience_counter >= config['training']['patience']:
            print(f'\nEarly stopping triggered after {epoch+1} epochs')
            break
    
    return history

def evaluate_model(model, test_loader, device, num_classes):
    """Evaluate the model on the test set with center pixel predictions."""
    model.eval()
    all_preds = []
    all_labels = []
    
    print("\nEvaluating model on test set...")
    with torch.no_grad():
        for data, labels in tqdm(test_loader):
            data = data.to(device)
            outputs = model(data)
            predictions = outputs.argmax(dim=1)
            
            predictions = predictions.cpu().numpy()
            labels = labels.numpy()
            
            valid_mask = (labels != 0)
            all_preds.extend(predictions[valid_mask])
            all_labels.extend(labels[valid_mask])
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    conf_matrix = confusion_matrix(all_labels, all_preds, labels=range(num_classes))
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        class_acc = np.zeros(num_classes)
        for i in range(num_classes):
            if conf_matrix[i].sum() > 0:
                class_acc[i] = conf_matrix[i, i] / conf_matrix[i].sum()
    
    overall_acc = accuracy_score(all_labels, all_preds)
    
    class_report = classification_report(all_labels, all_preds, 
                                      labels=range(num_classes),
                                      zero_division=0,
                                      output_dict=True)
    
    results = {
        'overall_accuracy': overall_acc,
        'class_accuracy': class_acc,
        'confusion_matrix': conf_matrix,
        'classification_report': class_report,
        'num_samples': len(all_labels)
    }
    
    return results

def fgsm_attack(data, epsilon, data_grad):
    """Generate FGSM attack."""
    # Create the perturbed image by adjusting each pixel of the input image
    perturbation = epsilon * data_grad.sign()
    perturbed_data = data + perturbation
    # Ensure data stays within [-3, 3] for standardized data
    perturbed_data = torch.clamp(perturbed_data, -3, 3)
    return perturbed_data

def calculate_adversarial_metrics(all_preds, all_clean_preds, all_labels, num_classes):
    """Calculate metrics for both clean and adversarial predictions."""
    all_preds = np.array(all_preds)
    all_clean_preds = np.array(all_clean_preds)
    all_labels = np.array(all_labels)
    
    clean_acc = accuracy_score(all_labels, all_clean_preds)
    adv_acc = accuracy_score(all_labels, all_preds)
    
    clean_conf_matrix = confusion_matrix(all_labels, all_clean_preds, labels=range(num_classes))
    adv_conf_matrix = confusion_matrix(all_labels, all_preds, labels=range(num_classes))
    
    clean_class_acc = np.zeros(num_classes)
    adv_class_acc = np.zeros(num_classes)
    
    for i in range(num_classes):
        if clean_conf_matrix[i].sum() > 0:
            clean_class_acc[i] = clean_conf_matrix[i, i] / clean_conf_matrix[i].sum()
        if adv_conf_matrix[i].sum() > 0:
            adv_class_acc[i] = adv_conf_matrix[i, i] / adv_conf_matrix[i].sum()
    
    return {
        'clean_accuracy': clean_acc,
        'adversarial_accuracy': adv_acc,
        'clean_class_accuracy': clean_class_acc,
        'adversarial_class_accuracy': adv_class_acc,
        'clean_confusion_matrix': clean_conf_matrix,
        'adversarial_confusion_matrix': adv_conf_matrix
    }

def print_adversarial_results(results, num_classes):
    """
    Print a detailed summary of adversarial evaluation results.
    
    Args:
        results: Dictionary containing adversarial evaluation metrics
        num_classes: Total number of classes in the dataset
    """
    print("\nAdversarial Evaluation Results Summary")
    print("=====================================")
    
    print("\nOverall Performance:")
    print(f"Clean Accuracy: {results['clean_accuracy']:.4f}")
    print(f"Adversarial Accuracy: {results['adversarial_accuracy']:.4f}")
    print(f"Robustness Gap: {results['clean_accuracy'] - results['adversarial_accuracy']:.4f}")
    # Calculate robustness score
    clean_correct = np.sum(np.diag(results['clean_confusion_matrix']))
    adv_correct = np.sum(np.diag(results['adversarial_confusion_matrix']))
    robustness_score = adv_correct / clean_correct if clean_correct > 0 else 0
    print(f"Robustness Score: {robustness_score:.4f}")
    
    print("\nPer-Class Performance:")
    print("---------------------")
    for i in range(1, num_classes):  # Skip background class (0)
        total_samples = results['clean_confusion_matrix'][i].sum()
        if total_samples > 0:
            clean_acc = results['clean_class_accuracy'][i]
            adv_acc = results['adversarial_class_accuracy'][i]
            acc_drop = clean_acc - adv_acc
            
            print(f"\nClass {i}:")
            print(f"  Total Samples: {total_samples}")
            print(f"  Clean Accuracy: {clean_acc:.4f}")
            print(f"  Adversarial Accuracy: {adv_acc:.4f}")
            print(f"  Accuracy Drop: {acc_drop:.4f}")
            
            # Calculate and display confusion rates
            clean_confusion = results['clean_confusion_matrix'][i]
            adv_confusion = results['adversarial_confusion_matrix'][i]
            
            if clean_confusion.sum() > 0:
                clean_misclass_rate = 1 - (clean_confusion[i] / clean_confusion.sum())
                print(f"  Clean Misclassification Rate: {clean_misclass_rate:.4f}")
            
            if adv_confusion.sum() > 0:
                adv_misclass_rate = 1 - (adv_confusion[i] / adv_confusion.sum())
                print(f"  Adversarial Misclassification Rate: {adv_misclass_rate:.4f}")
    
    print("\nOverall Statistics:")
    print("-----------------")
    total_samples = np.sum(results['clean_confusion_matrix'])
    
    print(f"Total Samples Evaluated: {total_samples}")
    print(f"Clean Correct Predictions: {clean_correct}")
    print(f"Adversarial Correct Predictions: {adv_correct}")
    print(f"Additional Misclassifications under Attack: {clean_correct - adv_correct}")

def evaluate_model_with_attacks(model, test_loader, device, num_classes, epsilon=0.01):
    """Evaluate model performance against adversarial attacks with center pixel predictions."""
    model.eval()
    all_preds = []
    all_labels = []
    all_clean_preds = []
    
    print("\nEvaluating model against FGSM attacks...")
    progress = tqdm(test_loader, desc='Testing with FGSM')
    
    for data, labels in progress:
        data, labels = data.to(device), labels.to(device)
        
        data.requires_grad = True
        outputs = model(data)
        
        valid_mask = (labels != 0)
        
        clean_predictions = outputs.argmax(dim=1)
        
        loss = F.cross_entropy(outputs[valid_mask], labels[valid_mask])
        loss.backward()
        
        perturbed_data = fgsm_attack(data, epsilon, data.grad.data)
        
        with torch.no_grad():
            adv_outputs = model(perturbed_data)
            predictions = adv_outputs.argmax(dim=1)
        
        predictions = predictions.cpu().numpy()
        clean_predictions = clean_predictions.cpu().numpy()
        labels = labels.cpu().numpy()
        
        valid_mask = labels != 0
        
        all_preds.extend(predictions[valid_mask])
        all_clean_preds.extend(clean_predictions[valid_mask])
        all_labels.extend(labels[valid_mask])
    
    results = calculate_adversarial_metrics(all_preds, all_clean_preds, all_labels, num_classes)
    print_adversarial_results(results, num_classes)
    
    return results

def plot_confusion_matrix(conf_matrix, class_names, save_path):
    """
    Plot confusion matrix with improved visualization.
    """
    plt.figure(figsize=(12, 10))
    
    # Normalize confusion matrix
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        norm_conf_matrix = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
        norm_conf_matrix = np.nan_to_num(norm_conf_matrix)  # Replace NaN with 0
    
    # Create heatmap
    sns.heatmap(norm_conf_matrix, 
                annot=conf_matrix,  # Show raw counts
                fmt='g',
                cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    
    plt.title('Confusion Matrix\n(colors show normalized values, numbers show counts)')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Save plot
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved confusion matrix plot to: {save_path}")

def plot_class_distribution(results, save_path):
    """
    Plot class distribution and accuracy.
    """
    class_samples = results['confusion_matrix'].sum(axis=1)
    class_acc = results['class_accuracy']
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot class distribution
    ax1.bar(range(len(class_samples)), class_samples)
    ax1.set_title('Class Distribution')
    ax1.set_xlabel('Class')
    ax1.set_ylabel('Number of Samples')
    
    # Plot class accuracy
    ax2.bar(range(len(class_acc)), class_acc)
    ax2.set_title('Class-wise Accuracy')
    ax2.set_xlabel('Class')
    ax2.set_ylabel('Accuracy')
    ax2.set_ylim([0, 1])
    
    # Save plot
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved class distribution plot to: {save_path}")

def main():
    # Setup
    setup_directories()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Data loading
    data_loader = HSIDataLoader(
        dataset_name=CONFIG['data']['dataset'],
        data_dir=CONFIG['data']['data_dir'],
        patch_size=CONFIG['data']['patch_size'],
        batch_size=CONFIG['data']['batch_size'],
        train_split=CONFIG['data']['train_size'],
        val_split=CONFIG['data']['val_size']
    )
    train_loader, val_loader, test_loader = data_loader.get_loaders()
    info = data_loader.get_dataset_info()
    
    # Model initialization
    model = SpectralFormer(
        num_spectral_bands=info['num_channels'],
        num_classes=info['num_classes'],
        patch_size=CONFIG['data']['patch_size']
    ).to(device)
    
    # Training
    history = train_model(model, train_loader, val_loader, device, CONFIG)
    plot_training_history(history, CONFIG['files']['training_plot'])
    
    # Testing
    # Load best model for testing
    best_model_path = os.path.join(CONFIG['paths']['checkpoints'], 'best_model.pth')
    model.load_state_dict(torch.load(best_model_path, weights_only=True))
    
    # Run both standard and adversarial evaluation
    print("\nStandard Evaluation:")
    test_metrics = evaluate_model(model, test_loader, device, info['num_classes'])

    print("\nAdversarial Evaluation:")
    adv_test_metrics = evaluate_model_with_attacks(model, test_loader, device, 
                                                info['num_classes'], epsilon=0.01)

    # Plot confusion matrices
    class_names = [f'Class {i}' for i in range(info['num_classes'])]
    plot_confusion_matrix(adv_test_metrics['clean_confusion_matrix'],
                        class_names,
                        os.path.join(CONFIG['paths']['results'], 'clean_confusion_matrix.png'))
    plot_confusion_matrix(adv_test_metrics['adversarial_confusion_matrix'],
                        class_names, 
                        os.path.join(CONFIG['paths']['results'], 'adversarial_confusion_matrix.png'))

if __name__ == "__main__":
    main()