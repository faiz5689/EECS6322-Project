import torch.nn as nn
import torch.nn.functional as F

class SimAM(nn.Module):
    def __init__(self, coeff_lambda=1e-4):
        super(SimAM, self).__init__()
        self.coeff_lambda = coeff_lambda

    def forward(self, X):
        """
        X: input tensor with shape (batch_size, num_channels, height, width)
        """
        assert X.dim() == 4, "shape of X must have 4 dimension"

        # spatial size
        n = X.shape[2] * X.shape[3] - 1
        n = 1 if n==0 else n

        # square of (t - u)
        d = (X - X.mean(dim=[2,3], keepdim=True)).pow(2)

        # d.sum() / n is channel variance
        v = d.sum(dim=[2,3], keepdim=True) / n

        # E_inv groups all importance of X
        E_inv = d / (4 * (v + self.coeff_lambda)) + 0.5

        # return attended features
        return X * F.sigmoid(E_inv)