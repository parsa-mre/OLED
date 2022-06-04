from tqdm import tqdm
from torch import nn
import torch.nn.functional as F
import torch
import config



def threshold(x):
    K = int(28 * 28 * (1 - config.T))
    b_s = x.shape[0]
    x = x.reshape(b_s, 28 * 28)
    _, frame = torch.topk(x, K)
    r = frame[:, -1]
    r = x[:, r]
    r = r.reshape(-1)
    r = r[::b_s+1]
    r = r.reshape(-1, 1)
    s = torch.add(-x, r)
    s = F.relu(s)
    s = s/(s + .0000000001)
    return s.reshape(b_s, 1, 28, 28)


def single_step(MM, R, dataloader, opt_MM, opt_R):
    
    L2 = nn.MSELoss()
    
    for x in dataloader:
        x = x.to(config.DEVICE)

        # Train Reconstructor
        mask = MM(x)
        mask = mask.detach()
        mask = threshold(mask)
        x_m = x * mask
        x_c = x * (torch.ones_like(mask) - mask)
        
        l_cont = L2(x_c, R(x_c))
        l_mask = L2(x, R(x_m)) ** 2
        l_rec = L2(x, R(x)) ** 2

        R_loss = l_mask + config.GAMMA * l_cont + config.LAMBDA * l_rec

        R.zero_grad()
        R_loss.backward()
        opt_R.step()


        # Train Mask Module 
        mask = MM(x)
        mask = threshold(mask)
        x_m = x * mask
        x_c = x * (torch.ones_like(mask) - mask)

        l_cont = L2(x_c, R(x_c))
        l_mask = L2(x, R(x_m)) ** 2

        MM_loss = -l_mask - config.GAMMA * l_cont

        MM.zero_grad()
        MM_loss.backward()
        opt_MM.step()

        print(f"  - mask loss: {MM_loss}")
        print(f"  - reconstructor loss: {R_loss}")

        return (MM_loss.detach(), R_loss.detach())







