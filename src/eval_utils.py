import torch
import torch.nn as nn
import numpy as np
import inspect
import dataset_utils

class DiscreteMMSE(nn.Module): #no-noise dMMSE
    def __init__(self, ws):
        super().__init__()
        self.register_buffer('ws', ws)
        self.N = ws.shape[0]
        self.dim = ws.shape[1]

    def forward(self, input_data):
        xs, ys = input_data
        ys = ys[:,:,0]
        B,t,d = xs.shape

        pred = torch.matmul(xs, self.ws.T)  # (batch_size, seq_len, num_weights)

        y_expanded = ys.unsqueeze(-1)  # (batch_size, seq_len, 1)

        squared_errors = (pred - y_expanded) ** 2 / t
        cumulative_sums = torch.cumsum(squared_errors, dim=1)  # (batch_size, seq_len, num_weights)

        min_indices = torch.argmin(cumulative_sums, dim=2)  # (batch_size, seq_len)

        preds = pred[torch.arange(B).unsqueeze(1), torch.arange(t).unsqueeze(0), min_indices]
        preds = preds.squeeze()

        return preds

def interpolate_great_circle_and_test(model, device, grad_idxs, criterion, epoch, dim, batch_size, ws,
                                      batches_per_epoch, seq_len, noise_std, offset=1000,
                                      num_interpolations=10, test_batches=50, lastonly=True, dmmse=None):
    assert ws.shape[0] >= 2, "At least two weight vectors are required for interpolation."

    l = ws.shape[0]
    idx1, idx2 = np.random.choice(l, size=2, replace=False)
    w1, w2 = ws[idx1], ws[idx2]

    dot_product = torch.dot(w1, w2)
    theta = torch.acos(dot_product)  # Angle in radians

    interp_weights = [
        (torch.sin((1 - alpha) * theta) * w1 + torch.sin(alpha * theta) * w2) / torch.sin(theta)
        for alpha in np.linspace(0, 1, num_interpolations)
    ]
    interp_losses = []
    dmmse_losses = []

    for idx, w in enumerate(interp_weights):
        print(f"Testing interpolation step {idx + 1}/{num_interpolations}")

        batch_ws = w.unsqueeze(0).repeat(batch_size, 1).to(device)

        loss = test_model(model, device, grad_idxs, criterion, epoch, dim, batch_size,
                          batches_per_epoch, seq_len, noise_std, offset=offset,
                          ws=batch_ws, lastonly=lastonly, test_batches=test_batches)
        if dmmse is not None:
            dmmse_idx = np.arange(seq_len)
            dmmse_loss = test_model(dmmse, device, dmmse_idx, criterion, epoch, dim, batch_size,
                                    batches_per_epoch, seq_len, noise_std, offset=offset,
                                    ws=batch_ws, lastonly=lastonly, test_batches=test_batches)
            dmmse_losses.append(dmmse_loss)
        interp_losses.append(loss)

    return interp_weights, interp_losses, dmmse_losses

def test_model(model, device, grad_idxs, criterion, epoch, dim, batch_size,
               batches_per_epoch, seq_len, noise_std, offset=1000,test_batches=50,
               ws=None,norm=False,lastonly=False):
    """Test the model on a set of weight vectors. 
    If ws is None, sample from the full sphere."""
    total_loss=0
    for b in range(test_batches):
        model.eval()
        #offset to ensure we don't test on the same data as training
        batch_seed = b + offset * batches_per_epoch * epoch
        xs, ys, ws = dataset_utils.gen_linreg_data(batch_seed,
                                                   batch_size=batch_size,dim=dim,
                                                   n_samples=seq_len,device=device,
                                                   noise_std=noise_std,
                                                   ws=ws,norm=norm)

        outputs = model((xs,ys))
        pred = outputs[:,grad_idxs].squeeze()
        true_ys = ys[:,:,0].squeeze()
        if lastonly:
            pred = pred[:,-1]
            true_ys = true_ys[:,-1]
        loss = criterion(pred,true_ys)

        total_loss += loss.item()

    return total_loss/test_batches

def test_cone_falloff(model, device, grad_idxs, criterion, epoch, dim, batch_size,
               batches_per_epoch, seq_len, noise_std, offset=1000,test_batches=10,
               start_angle=0, end_angle=180, strip_width=5, gaussianize=True, lastonly=False, **kwargs):
    """Test the model's performance across a range of angles."""
    angles = []
    losses = []
    strip_width_ = strip_width * np.pi/180
    for a in range(start_angle, end_angle, strip_width):
        a_ = a * np.pi/180
        ws = dataset_utils.sample_cone(batch_size, dim, max_theta=a_+strip_width_, 
                                       min_theta=a_, gaussianize=gaussianize)
        loss = test_model(model, device, grad_idxs, criterion, epoch, dim, batch_size,
               batches_per_epoch, seq_len, noise_std, offset=offset,test_batches=test_batches,
               ws=ws,lastonly=lastonly)
        angles.append(a)
        losses.append(loss)
    return angles, losses