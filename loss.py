import torch
import torch.nn as nn
import torch.nn.functional as F

class HSIAdvLoss(nn.Module):
    def __init__(self, model, epsilon=0.01, alpha=0.1):
        super(HSIAdvLoss, self).__init__()
        self.model = model
        self.epsilon = epsilon
        self.alpha = alpha

    def generate_adversarial_examples(self, data, labels):
        """
        Generate adversarial examples using FGSM for center pixel classification.
        """
        # Create a copy that requires gradients
        perturbed_data = data.detach().clone()
        perturbed_data.requires_grad = True
        
        # Forward pass
        outputs = self.model(perturbed_data)
        
        # Get valid mask for center pixels
        valid_mask = (labels != 0)
        
        if valid_mask.sum() == 0:
            return data
        
        # Calculate loss for valid pixels only
        loss = F.cross_entropy(
            outputs[valid_mask],
            labels[valid_mask],
            label_smoothing=0.1
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
        # Clamp to [-3, 3] range for standardized data
        perturbed_data = torch.clamp(perturbed_data, -3, 3)
        
        return perturbed_data

    def forward(self, data, labels, training=True):
        """
        Forward pass computing standard loss for center pixel classification.
        Only computes adversarial loss if alpha > 0 and in training mode.
        """
        metrics = {}
        
        # Standard forward pass
        outputs = self.model(data)
        
        # Get valid mask (non-zero labels)
        valid_mask = (labels != 0)
        
        # Compute standard loss and accuracy for valid pixels
        if valid_mask.sum() > 0:
            standard_loss = F.cross_entropy(
                outputs[valid_mask],
                labels[valid_mask],
                label_smoothing=0.1
            )
            
            # Calculate accuracy
            predictions = outputs.argmax(dim=1)
            correct_pixels = (predictions == labels) & valid_mask
            accuracy = correct_pixels.float().mean()
        else:
            standard_loss = torch.tensor(0.0, device=data.device)
            accuracy = torch.tensor(0.0, device=data.device)
        
        metrics['standard_loss'] = standard_loss.item()
        metrics['accuracy'] = accuracy.item()
        
        if training and self.alpha > 0:
            if valid_mask.sum() > 0:
                # Generate and evaluate adversarial examples
                with torch.set_grad_enabled(True):
                    perturbed_data = self.generate_adversarial_examples(data, labels)
                    adv_outputs = self.model(perturbed_data)
                    
                    # Compute adversarial loss
                    adv_loss = F.cross_entropy(
                        adv_outputs[valid_mask],
                        labels[valid_mask],
                        label_smoothing=0.1
                    )
                    
                    # Calculate adversarial accuracy
                    adv_predictions = adv_outputs.argmax(dim=1)
                    correct_adv_pixels = (adv_predictions == labels) & valid_mask
                    adv_accuracy = correct_adv_pixels.float().mean()
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
        metrics['valid_pixels'] = valid_mask.sum().item()
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