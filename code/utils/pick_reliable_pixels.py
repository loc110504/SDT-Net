import torch

def refine_high_confidence(pred_logits, threshold=0.5):
    # Get the predicted class (highest probability) for each pixel
    pred_class = torch.argmax(pred_logits, dim=1)
    # Extract probability maps for each class
    prob_cls0 = pred_logits[:, 0, :, :]
    prob_cls1 = pred_logits[:, 1, :, :]
    prob_cls2 = pred_logits[:, 2, :, :]
    prob_cls3 = pred_logits[:, 3, :, :]
    # Generate high-confidence masks for each class
    mask_cls0 = torch.where((prob_cls0 > threshold) & (pred_class == 0), 5, 0)
    mask_cls1 = torch.where((prob_cls1 > threshold) & (pred_class == 1), 1, 0)
    mask_cls2 = torch.where((prob_cls2 > threshold) & (pred_class == 2), 2, 0)
    mask_cls3 = torch.where((prob_cls3 > threshold) & (pred_class == 3), 3, 0)
    
    # Identify confident and non-confident pixels
    high_conf_mask = ((mask_cls0 + mask_cls1 + mask_cls2 + mask_cls3) > 0).to(torch.int32)
    low_conf_mask = torch.ones_like(high_conf_mask) - high_conf_mask
    
    # Combine all class masks and assign 4 to uncertain regions
    refined_labels = mask_cls0 + mask_cls1 + mask_cls2 + mask_cls3 + 4 * low_conf_mask
    refined_labels[refined_labels == 5] = 0
    
    return refined_labels

