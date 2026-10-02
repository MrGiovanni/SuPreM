import torch
import torch.nn.functional as F
import torch.nn as nn

class BinaryDiceLoss(nn.Module):
    """
    Calculates the Binary Dice Loss (Sørensen-Dice coefficient) for binary segmentation.  
    """
    def __init__(self, smooth=1, p=2, reduction='mean'):
        """
        Args:
            smooth (float, default=1): Smoothing factor for numerical stability.
            p (int, default=2): Exponent in the denominator of the Dice coefficient.
            reduction (str, default='mean'): Specifies the reduction to apply to the loss:
                'mean': Averages the loss over the batch.
                'sum': Sums the loss over the batch.
                'none': Returns the loss per image without reduction.
        """
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction

    def forward(self, predict, target):
        """
        Calculates the Binary Dice Loss.

        Args:
            predict (Tensor): Model predictions (B, C, H, W, D)
            target (Tensor): Ground truth masks (B, C, H, W, D)

        Returns:
            Tensor: The scalar Binary Dice Loss value.
        """
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        num = torch.sum(torch.mul(predict, target), dim=1)
        den = torch.sum(predict, dim=1) + torch.sum(target, dim=1) + self.smooth

        dice_score = 2*num / den
        dice_loss = 1 - dice_score
        
        dice_loss_avg = dice_loss.sum() / dice_loss.shape[0]

        return dice_loss_avg

class DiceLoss(nn.Module):
    """
    Computes Dice Loss for multi-class segmentation, averaging losses across classes.
    Designed for sigmoid outputs from segmentation models. 
    """
    def __init__(self, weight=None, ignore_index=None, num_classes=3, **kwargs):
        """
        Args:
            weight (Tensor, optional): Class weights for potential imbalance.
            ignore_index (int, optional): Class index to ignore in loss calculation.
            num_classes (int, default=3): The number of classes.
            **kwargs: Additional arguments passed to BinaryDiceLoss.
        """
        super(DiceLoss, self).__init__()
        self.kwargs = kwargs
        self.weight = weight
        self.ignore_index = ignore_index
        self.num_classes = num_classes
        self.dice = BinaryDiceLoss(**self.kwargs)

    def forward(self, predict, target):
        """
        Computes the multi-class Dice Loss.

        Args:
            predict (Tensor): Model predictions (B, C, H, W, D)
            target (Tensor): Ground truth masks (B, C, H, W, D)

        Returns:
            Tensor: The scalar average Dice Loss across all classes.
        """
        predict = F.sigmoid(predict)
        B = predict.shape[0]
        
        # Compute target presence mask efficiently using sum across spatial dimensions
        target_sum = torch.sum(target, dim=(2, 3, 4))  # Shape: (B, C)
        assert target_sum.shape[1] == self.num_classes, f'Number of target classes {target_sum.shape[1]} does not match expected {self.num_classes}'
        
        # Create mask of present organs (non-zero targets)
        present_mask = target_sum > 0  # Shape: (B, C)
        
        # If no organs present, return default loss
        if not present_mask.any():
            return torch.tensor(1.0, device=predict.device)
        
        # Compute dice loss for all present organs at once
        total_loss = []
        for b in range(B):
            present_organs = torch.nonzero(present_mask[b], as_tuple=True)[0]
            if len(present_organs) > 0:
                for organ in present_organs:
                    dice_loss = self.dice(predict[b, organ], target[b, organ])
                    total_loss.append(dice_loss)
        
        if len(total_loss) == 0:
            return torch.tensor(1.0, device=predict.device)
        
        total_loss = torch.stack(total_loss)
        return total_loss.mean()

class Multi_BCELoss(nn.Module):
    """
    Calculates multi-class Binary Cross Entropy (BCE) Loss, averaging losses across classes.
    """
    def __init__(self, ignore_index=None, num_classes=3, **kwargs):
        """
        Initializes the Multi_BCELoss object.

        Args:
            ignore_index (int, optional): Class index to ignore in loss calculation.
            num_classes (int, default=3): The number of classes.
            **kwargs: Additional arguments for the underlying BCEWithLogitsLoss.
        """
        super(Multi_BCELoss, self).__init__()
        self.kwargs = kwargs
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, predict, target):
        """
        Computes the multi-class BCE Loss.

        Args:
            predict (Tensor): Model predictions (B, C, H, W, D)
            target (Tensor): Ground truth masks (B, C, H, W, D)

        Returns:
            Tensor: The scalar average BCE Loss across all classes.
        """
        assert predict.shape[2:] == target.shape[2:], 'predict & target shape do not match'
        
        # Reshape for efficient computation: (B*C, H*W*D)
        B, C = predict.shape[:2]
        predict_flat = predict.reshape(B * C, -1)
        target_flat = target.reshape(B * C, -1)
        
        # Compute BCE loss for all batch-class combinations at once
        ce_loss = F.binary_cross_entropy_with_logits(predict_flat, target_flat, reduction='none')
        # Average over spatial dimensions, then over batch and classes
        ce_loss = ce_loss.mean(dim=1).mean()
        
        return ce_loss