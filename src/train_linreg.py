import torch
import numpy as np
import matplotlib.pyplot as plt
import transformer
import dataset_utils
import time
from pathlib import Path
from eval_utils import DiscreteMMSE, interpolate_great_circle_and_test
from eval_utils import test_model, test_cone_falloff

def get_batch(pretrain_size, batch_size, ws=None, online=False, dim=10, angle=np.pi, min_angle=0., gaussianize=False):
    if online:
        return dataset_utils.sample_cone(batch_size, dim, angle, min_theta=min_angle, gaussianize=gaussianize)

    if pretrain_size >= batch_size:
        batch_ws_idxs = np.random.permutation(pretrain_size)[:batch_size]
    else:
        N = batch_size // pretrain_size
        batch_ws_idxs = np.random.permutation(pretrain_size)
        for i in range(N-1):
            addl = np.random.permutation(pretrain_size)
            batch_ws_idxs = np.concatenate((batch_ws_idxs,addl))
        remaining_size = batch_size % pretrain_size
        addl = np.random.permutation(pretrain_size)[:remaining_size]
        batch_ws_idxs = np.concatenate((batch_ws_idxs,addl))
    return ws[batch_ws_idxs]

def train(batch_size=128, lr=3e-4, epochs=120, batches_per_epoch=100, device='cuda',
          seq_len=50, d_model=128, n_layer=10, dim=10, noise_std=0, checkpoint_dir="checkpoints/", 
          pretrain_size=2**10, angle=180, lr_milestones=[], gaussianize=False,
          save_freq=3, output_dir="output/"):
    """Train a transformer to do in-context linear regression with weight vectors sampled
    from a hyperspherical cap with a given angle.
    Args:
        batch_size (int): Size of each training batch.
        lr (float): Initial learning rate for the optimizer.
        epochs (int): Number of training epochs.
        batches_per_epoch (int): Number of batches per epoch.
        device (str): Device to run the training on ('cuda' or 'cpu').
        seq_len (int): Number of (x,y) regression pairs in the input sequences.
        d_model (int): Dimension of the transformer state.
        n_layer (int): Number of layers in the transformer.
        dim (int): Dimension of the input data.
        noise_std (float): Standard deviation of the label noise added to the input data.
        checkpoint_dir (str): Output directory for saving the model and results.
        pretrain_size (int): Number of pretraining weight vectors to use.
        angle (float): Half-angle of the hyperspherical cap in degrees.
        lr_milestones (list): List of epochs at which to reduce the learning rate, if any.
        gaussianize (bool): Whether to apply Gaussianization to the weight vectors. If False,
            the weight vectors will have norm 1
        save_freq (int): Frequency (in epochs) with which to save model checkpoints.
        output_dir (str): Output directory for saving the model and results.
    Returns:
        model (torch.nn.Module): The trained transformer model.
        loss_history (np.ndarray): Array of training loss values over time.
        test_loss_history (np.ndarray): Array of test loss values over epochs. 
            *Note that the test loss is computed over the entire hypersphere during training.*
        angles (np.ndarray): Array of angles used in the OOD falloff evaluation.
        losses (np.ndarray): Array of losses corresponding to the angles in the OOD evaluation
    """

    print("Training on half-angle: ", angle)
    angle_degrees = angle
    angle = angle * np.pi/180

    #create output directores if they don't exist
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    model = transformer.RegressionTransformer(d_model=d_model,device=device,
                                            block_size=seq_len*2,n_layer=n_layer,
                                            input_dim=dim)
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                        milestones=lr_milestones)

    # Output indices for which we want to compute the gradient 
    grad_idxs = np.array(list(range(0,2*seq_len,2)))

    #Store training and test loss history (Note that test loss is computed over the *entire* hypersphere during training)
    loss_history = [0]*batches_per_epoch*epochs
    test_loss_history = [0]*epochs

    ws = dataset_utils.sample_cone(pretrain_size, dim, angle, gaussianize=gaussianize) 
    dmmse = DiscreteMMSE(ws)

    for epoch in range(1,epochs+1):
        print("Epoch: ", epoch)
        t1 = time.time()

        model.train()
        for b in range(batches_per_epoch):
            batch_ws = get_batch(pretrain_size, batch_size, ws)           
            batch_seed = b+batches_per_epoch*epoch
            xs, ys, _ = dataset_utils.gen_linreg_data(batch_seed,batch_size=batch_size,
                                                    dim=dim,n_samples=seq_len,device=device,noise_std=noise_std,
                                                    ws=batch_ws)
            optimizer.zero_grad()
            outputs = model((xs,ys))
            #only take the gradients of the output indices we care about (subseqs that haven't seen labels)
            pred = outputs[:,grad_idxs].squeeze()
            true_ys = ys[:,:,0].squeeze()
            loss = criterion(pred,true_ys)
            loss.backward()
            optimizer.step()
            loss_history[b+batches_per_epoch*(epoch-1)] = loss.item()

        if (epoch - 1) % save_freq == 0:
            torch.save(model.state_dict(),checkpoint_dir + "checkpoint{0}.th".format(epoch))

        test_loss = test_model(model, device, grad_idxs, criterion, epoch, dim, batch_size,
                               batches_per_epoch, seq_len, noise_std, offset=1000*epoch,
                               norm=(not gaussianize))
        test_loss_history[epoch-1] = test_loss

        lr_scheduler.step()
        t2 = time.time()
        print("Epoch complete, time:", (t2-t1)/60, "minutes, loss: {0}, test loss: {1}".format(loss.item(),test_loss))

    loss_history = np.array(loss_history)

    torch.save(model.state_dict(), checkpoint_dir + "final.th")
    np.save(checkpoint_dir + "loss_history.npy", loss_history)
    np.save(checkpoint_dir + "test_loss_history.npy", test_loss_history)

    angles, losses = test_cone_falloff(model, device, grad_idxs, criterion, epoch, dim, batch_size,
               batches_per_epoch, seq_len, noise_std, offset=10000,test_batches=250,
               start_angle=0, end_angle=180, strip_width=5, gaussianize=gaussianize,lastonly=True)

    interp_weights, interp_losses, dmmse_losses = interpolate_great_circle_and_test(model, device, grad_idxs, criterion, epoch, dim, batch_size,
                                      ws, batches_per_epoch, seq_len, noise_std, dmmse=dmmse)

    task = int(np.log2(pretrain_size))

    np.save(output_dir+"interp_losses_angle{0}_tasks{1}".format(angle_degrees,task),interp_losses)
    np.save(output_dir+"interp_dmmse_angle{0}_tasks{1}".format(angle_degrees,task),dmmse_losses)

    np.save(output_dir+"losses_angle{0}_tasks{1}.npy".format(angle_degrees,task), losses)
    return model, loss_history, test_loss_history, angles, losses
