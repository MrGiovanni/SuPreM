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
        total_loss = []
        predict = F.sigmoid(predict)
        B = predict.shape[0]

        for b in range(B):
            target_sum = torch.sum(target[b], axis = (1,2,3))
            # print('target_sum:',target_sum)
            assert len(target_sum) == self.num_classes, 'target sum =! 25 (25 is set by default for args.num_class in train.py)'
            non_zero_tensor = torch.nonzero(target_sum).squeeze()
            non_zero_list = non_zero_tensor.tolist() if non_zero_tensor.dim() > 0 else [non_zero_tensor.tolist()]
            # print('non_zero_list:',non_zero_list)
            # by default, the num_classes is 25, could be changed in train.py
            organ_list = [i for i in range(self.num_classes)]
            # print('organ_list:',organ_list)
            new_list = []
            for idx in non_zero_list:
                if idx in organ_list:
                    new_list.append(idx)
                    # print('new_list:',new_list)
            if len(new_list)!=0:     
                for organ in new_list:
                    dice_loss = self.dice(predict[b, organ], target[b, organ])
                    total_loss.append(dice_loss)
        
        if len(total_loss) == 0:
            return torch.tensor(1.0).cuda()
        total_loss = torch.stack(total_loss)

        return total_loss.sum()/total_loss.shape[0]

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

        total_loss = []
        B = predict.shape[0]
        # print('predict shape:',predict.shape)
        # print('target shape:',target.shape)

        for b in range(B):
            for organ in range(self.num_classes):
                ce_loss = self.criterion(predict[b, organ], target[b, organ])
                total_loss.append(ce_loss)
        total_loss = torch.stack(total_loss)
        return total_loss.sum()/total_loss.shape[0]