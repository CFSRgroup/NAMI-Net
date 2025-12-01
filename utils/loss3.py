import torch
from torch import nn
from torch.autograd.function import Function
import torch.nn.functional as F
# References & Acknowledgments
# Portions of the code in this file were adapted from the following source:
# https://github.com/CFSRgroup/Paozival  https://github.com/dvlab-research/  https://github.com/declare-lab/MISA

class CenterLoss(nn.Module):
    def __init__(self, num_classes, feat_dim, size_average=True, device='cpu'):
        super(CenterLoss, self).__init__()
        self.device = device
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim).to(device))
        self.centerloss_func = CenterLossFunction.apply
        self.feat_dim = feat_dim
        self.size_average = size_average

    def forward(self, label, feat):
        label = label.to(self.device)
        feat = feat.to(self.device)

        if feat.size(1) != self.feat_dim:
            raise ValueError(
                "Center's dim: {0} should match input feature's dim: {1}".format(self.feat_dim, feat.size(1)))

        batch_size_tensor = feat.new_empty(1).fill_(feat.size(0) if self.size_average else 1)
        center_loss = self.centerloss_func(feat, label, self.centers, batch_size_tensor)

        inter_class_loss = InterClassLoss(self.centers.to(self.device))()
        similarity_loss = SimilarityLoss(order=4)(feat, label)
        difference_loss = DifferenceLoss()(feat, label)

        center_loss = ( center_loss / (1 + center_loss) )*0.1
        inter_class_loss = inter_class_loss / (1 + inter_class_loss)
        similarity_loss = similarity_loss / (1 + similarity_loss)
        difference_loss = difference_loss / (1 + difference_loss)

        total_loss = center_loss + inter_class_loss + similarity_loss + difference_loss
        return total_loss.to(self.device)


class CenterLossFunction(Function):
    @staticmethod
    def forward(ctx, feature, label, centers, batch_size):
        feature, label = feature.to(centers.device), label.to(centers.device)
        ctx.save_for_backward(feature, label, centers, batch_size)

        centers_batch = centers.index_select(0, label.long())
        return (feature - centers_batch).pow(2).sum() / 2.0 / batch_size

    @staticmethod
    def backward(ctx, grad_output):
        feature, label, centers, batch_size = ctx.saved_tensors
        centers_batch = centers.index_select(0, label.long())
        diff = centers_batch - feature

        counts = centers.new_ones(centers.size(0))
        ones = centers.new_ones(label.size(0))
        grad_centers = centers.new_zeros(centers.size()).to(centers.device)

        counts = counts.scatter_add_(0, label.long(), ones)
        grad_centers.scatter_add_(0, label.unsqueeze(1).expand(feature.size()).long(), diff)
        grad_centers = grad_centers / counts.view(-1, 1)
        return -grad_output * diff / batch_size, None, grad_centers / batch_size, None


class InterClassLoss:
    def __init__(self, centers, epsilon=1e-6):
        self.centers = centers
        self.epsilon = epsilon

    def __call__(self):
        center_diff = self.centers.unsqueeze(1) - self.centers.unsqueeze(0)
        dist_matrix = torch.norm(center_diff, dim=2) + self.epsilon
        mask = torch.triu(torch.ones_like(dist_matrix), diagonal=1)
        loss = (1.0 / (dist_matrix ** 2) * mask).sum()
        return loss.to(self.centers.device)

class SimilarityLoss:
    def __init__(self, order=4):
        self.order = order

    def __call__(self, feat, label):
        loss = 0.0
        total_pairs = 0

        unique_labels = torch.unique(label)
        for lbl in unique_labels:
            mask = label == lbl
            feat_same_class = feat[mask]

            if feat_same_class.size(0) < 2:
                continue

            min_val, _ = torch.min(feat_same_class, dim=0)
            max_val, _ = torch.max(feat_same_class, dim=0)
            interval = torch.abs(max_val - min_val).clamp(min=1e-6)
            regularizer = 1.0 / (
                interval.mean().pow(torch.arange(1, self.order + 1, dtype=torch.float32).to(feat.device)))

            indices = torch.combinations(torch.arange(feat_same_class.size(0)), r=2).to(feat.device)
            num_pairs = indices.size(0)

            total_pairs += num_pairs
            class_loss = 0.0

            feat_pairs = feat_same_class[indices]

            for k in range(1, self.order + 1):
                center_moments = (feat_pairs - feat_pairs.mean(dim=1, keepdim=True)).pow(k).mean(dim=1)
                moment_diff = (center_moments[:, 0] - center_moments[:, 1]).pow(2)
                class_loss += (regularizer[k - 1] * moment_diff).sum()

            loss += class_loss / num_pairs

        return loss / len(unique_labels) if total_pairs > 0 else loss



class DifferenceLoss:
    def __call__(self, feat, label):
        batch_size = feat.size(0)
        feat = feat.view(batch_size, -1)

        feat_mean = torch.mean(feat, dim=0, keepdims=True)
        feat = feat - feat_mean
        feat_norm = torch.norm(feat, p=2, dim=1, keepdim=True).detach()
        feat_l2 = feat.div(feat_norm.expand_as(feat) + 1e-6)

        mask_diff_class = (label.unsqueeze(1) != label.unsqueeze(0)).float()
        dot_products = torch.mm(feat_l2, feat_l2.t()).pow(2)
        diff_loss = (dot_products * mask_diff_class).sum() / mask_diff_class.sum()

        return diff_loss

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_classes = 5
    feat_dim = 128

    features = torch.randn(5, feat_dim).to(device)
    labels = torch.tensor([0, 1, 2, 1, 2]).to(device)

    center_loss = CenterLoss(num_classes=num_classes, feat_dim=feat_dim, device=device)

    total_loss = center_loss(labels, features)
    print(f"Total Loss: {total_loss.item():.4f}")

class PaCoLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=1.0, gamma=0.0, supt=1.0, temperature=1.0, base_temperature=None, num_classes=3):
        super(PaCoLoss, self).__init__()
        self.temperature = temperature
        self.base_temperature = temperature if base_temperature is None else base_temperature
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.supt = supt
        self.num_classes = num_classes

    def forward(self, features, labels, sup_logits):
        device = features.device
        batch_size = features.shape[0]

        labels = labels.contiguous().view(-1, 1).long()

        centers_similarity = sup_logits / self.supt

        sample_similarity = torch.matmul(features, features.T) / self.temperature

        anchor_dot_contrast = torch.cat((centers_similarity, sample_similarity), dim=1)

        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        logits_mask = torch.ones_like(sample_similarity, device=device).scatter_(1, torch.arange(batch_size).view(-1, 1).to(device), 0)

        one_hot_label = F.one_hot(labels.view(-1), num_classes=self.num_classes).float().to(device)

        mask = torch.cat((one_hot_label * self.beta, logits_mask * self.alpha), dim=1)

        logits_mask = torch.cat((torch.ones(batch_size, self.num_classes, device=device), self.gamma * logits_mask), dim=1)
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)

        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()

        return loss



