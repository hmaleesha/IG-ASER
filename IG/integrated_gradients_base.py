import torch
import numpy as np

def integrated_gradients(model, input_tensor, baseline, target_class, steps):
    """
    Computes Integrated Gradients using the Trapezoidal rule for a PyTorch model.
    Args:
        model: The PyTorch model.
        input_tensor: The input tensor (e.g., MFCCs) for which IG is computed.
        baseline: A baseline input tensor (default is zero tensor of the same shape as input_tensor).
        target_class: Index of the target class.
        steps: Number of interpolation steps between baseline and input.
    Returns:
        attributions: IG attributions for the input.
    """
    input_tensor.requires_grad_()
    
    # Scale inputs and compute scaled inputs
    scaled_inputs = [
        baseline + (float(i) / steps) * (input_tensor - baseline) for i in range(steps + 1)
    ]
    scaled_inputs = torch.stack(scaled_inputs)
    
    # Initialize gradient accumulator
    total_gradients = torch.zeros_like(input_tensor).to(input_tensor.device)
    prev_gradients = None
    
    for scaled_input in scaled_inputs:
        # Forward pass
        outputs = model(scaled_input.unsqueeze(0))  # Add batch dimension if needed
        
        # Handle tuple outputs
        if isinstance(outputs, tuple):
            outputs = outputs[0]  # Adjust index if needed
        
        # Select specific class for classification
        if target_class is not None:
            outputs = outputs[:, target_class]
        else:
            outputs = outputs.sum()  # Aggregate for regression or multi-output tasks
        
        if outputs.ndimension() > 1:
            outputs = outputs.mean()
        
        # Backward pass
        gradients = torch.autograd.grad(outputs, input_tensor, retain_graph=True)[0]
        
        if prev_gradients is not None:
            total_gradients += (prev_gradients + gradients) / 2  # Trapezoidal rule
        
        prev_gradients = gradients.clone()
    
    # Average gradients and compute attributions
    avg_gradients = total_gradients / steps
    attributions = (input_tensor - baseline) * avg_gradients
    return attributions