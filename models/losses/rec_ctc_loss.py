import torch
from torch import nn


class CTCLoss(nn.Module):

    def __init__(self, use_focal_loss=False, **kwargs):
        super(CTCLoss, self).__init__()
        self.loss_func = nn.CTCLoss(blank=0, reduction='none')
        self.use_focal_loss = use_focal_loss

    def forward(self, predicts, batch):
        if isinstance(predicts, (list, tuple)):
            predicts = predicts[-1]
        predicts = predicts.permute(1, 0, 2)
        N, B, _ = predicts.shape
        preds_lengths = torch.tensor([N] * B).long()
        labels = batch[1].int()
        label_lengths = batch[2].long()
        ## different between paddle and torch, paddle has a softmax op in ctc_loss, but torch has not
        loss = self.loss_func(predicts.log_softmax(2), labels, preds_lengths, label_lengths)
        if self.use_focal_loss:
            weight = torch.exp(-loss)
            weight = torch.subtract(torch.tensor([1.0]), weight)
            weight = torch.square(weight)
            loss = torch.mul(loss, weight)
        loss = loss.mean()
        return {'loss': loss}

if __name__=="__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    import numpy as np
    np.random.seed(1)
    predicts = np.random.randn(128, 40, 176).astype(np.float32)
    batch = [
        np.random.randn(128, 3, 48, 320).astype(np.float32),
        np.random.randn(128, 25).astype(np.float32),
        np.ones(128).astype(np.float32),
        np.ones(128).astype(np.float32),
    ]
    predicts = torch.from_numpy(predicts).to(device) # 40, 128, 176
    batch = [torch.from_numpy(item).to(device) for item in batch]
    loss = CTCLoss()
    # 128, 40, 176; [128, 3, 48, 320; 128, 25; 128; 128]
    out = loss(predicts, batch)['loss']
    print(out.cpu().numpy())
    # print('==> ',np.sum(out), np.mean(out), np.max(out), np.min(out))
    