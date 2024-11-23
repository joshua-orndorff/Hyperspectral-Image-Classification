import torch
import torch.nn as nn
import torch.nn.functional as F

class HSIAdvLoss(nn.Module):
    def __init__(self, model, epsilon=0.01, alpha=0.1):
        super(HSIAdvLoss, self).__init__()
        self.model = model
        self.epsilon = epsilon
        self.alpha = alpha

    def generate_adversarial_examples(self, data, label_patches):
        """
        Generate adversarial examples using FGSM.
        """
        # Create a copy that requires gradients
        perturbed_data = data.detach().clone()
        perturbed_data.requires_grad = True
        
        # Forward pass
        outputs = self.model(perturbed_data)
        
        # Reshape outputs to match label patches
        batch_size = outputs.size(0)
        num_classes = outputs.size(1)
        patch_size = label_patches.size(1)
        outputs = outputs.view(batch_size, num_classes, patch_size, patch_size)
        
        # Get valid mask for all pixels
        valid_mask = (label_patches != 0)
        
        if valid_mask.sum() == 0:
            return data
        
        # Calculate loss for valid pixels only
        loss = F.cross_entropy(
            outputs.permute(0, 2, 3, 1).reshape(-1, num_classes),
            label_patches.reshape(-1),
            ignore_index=0  # Ignore padded pixels
        )
        
        # Calculate gradients
        if perturbed_data.grad is not None:
            perturbed_data.grad.zero_()
        loss.backward()
        
        # Create perturbation
        data_grad = perturbed_data.grad.detach()
        perturbation = self.epsilon * data_grad.sign()
        
        # Generate adversarial example
        perturbed_data = data.detach() + perturbation
        perturbed_data = torch.clamp(perturbed_data, data.min(), data.max())
        
        return perturbed_data

    def forward(self, data, label_patches, training=True):
        """
        Forward pass computing both standard and adversarial loss.
        """
        metrics = {}
        
        # Standard forward pass
        outputs = self.model(data)
        
        # Reshape outputs to match label patches
        batch_size = outputs.size(0)
        num_classes = outputs.size(1)
        patch_size = label_patches.size(1)
        outputs = outputs.view(batch_size, num_classes, patch_size, patch_size)
        
        # Get valid mask (non-zero labels)
        valid_mask = (label_patches != 0)
        
        # Compute standard loss and accuracy for all valid pixels
        if valid_mask.sum() > 0:
            standard_loss = F.cross_entropy(
                outputs.permute(0, 2, 3, 1).reshape(-1, num_classes),
                label_patches.reshape(-1),
                ignore_index=0,
                label_smoothing=0.1
            )
            
            # Calculate pixel-wise accuracy
            predictions = outputs.argmax(dim=1)
            correct_pixels = (predictions == label_patches) & valid_mask
            accuracy = correct_pixels.float().sum() / valid_mask.float().sum()
        else:
            standard_loss = torch.tensor(0.0, device=data.device)
            accuracy = torch.tensor(0.0, device=data.device)
        
        metrics['standard_loss'] = standard_loss.item()
        metrics['accuracy'] = accuracy.item()
        
        if training:
            if valid_mask.sum() > 0:
                # Generate and evaluate adversarial examples
                with torch.set_grad_enabled(True):
                    perturbed_data = self.generate_adversarial_examples(data, label_patches)
                    adv_outputs = self.model(perturbed_data)
                    adv_outputs = adv_outputs.view(batch_size, num_classes, patch_size, patch_size)
                    
                    # Compute adversarial loss
                    adv_loss = F.cross_entropy(
                        adv_outputs.permute(0, 2, 3, 1).reshape(-1, num_classes),
                        label_patches.reshape(-1),
                        ignore_index=0
                    )
                    
                    # Calculate adversarial accuracy
                    adv_predictions = adv_outputs.argmax(dim=1)
                    correct_adv_pixels = (adv_predictions == label_patches) & valid_mask
                    adv_accuracy = correct_adv_pixels.float().sum() / valid_mask.float().sum()
            else:
                adv_loss = torch.tensor(0.0, device=data.device)
                adv_accuracy = torch.tensor(0.0, device=data.device)
            
            # Compute total loss with reduced adversarial component
            total_loss = (1 - self.alpha) * standard_loss + self.alpha * adv_loss
            
            metrics['adversarial_loss'] = adv_loss.item()
            metrics['adversarial_accuracy'] = adv_accuracy.item()
            metrics['total_loss'] = total_loss.item()
            metrics['valid_pixels'] = valid_mask.sum().item()
            
            return total_loss, metrics
        
        metrics['total_loss'] = standard_loss.item()
        return standard_loss, metrics

    def get_metrics_string(self, metrics):
        """Format metrics into a readable string."""
        metrics_str = f"Loss: {metrics['standard_loss']:.4f}"
        metrics_str += f" | Acc: {metrics['accuracy']:.4f}"
        if 'adversarial_loss' in metrics:
            metrics_str += f" | Adv_Loss: {metrics['adversarial_loss']:.4f}"
            metrics_str += f" | Adv_Acc: {metrics['adversarial_accuracy']:.4f}"
        if 'valid_pixels' in metrics:
            metrics_str += f" | Valid Pixels: {metrics['valid_pixels']}"
        return metrics_str