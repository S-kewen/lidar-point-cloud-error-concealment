from models.utils import chamfer_loss, EMD
import torch
import numpy as np
import torch
from geomloss import SamplesLoss
def batch_EMD_loss(x, y):
    bs, num_points_x, points_dim = x.size()
    _, num_points_y, _ = y.size()
    batch_EMD = 0
    L = SamplesLoss(loss = 'gaussian', p=2, blur=.00005)
    for i in range(bs):
        loss = L(x[i], y[i])
        batch_EMD += loss
    emd = batch_EMD/bs
    return emd

def main():
    pc1 = np.fromfile("/mnt/sdc/exp_random_packet_loss_rate/receiver_none_0.2/1000.bin", dtype=np.float32, count=-1).reshape([-1, 4])
    pc2 = np.fromfile("/mnt/sdc/exp_random_packet_loss_rate/receiver_none_0.2/1000.bin", dtype=np.float32, count=-1).reshape([-1, 4]).cuda(non_blocking=True)
    emd = EMD(pc1, pc2)
    emd = emd.squeeze().cpu().numpy()
    print(emd)



if __name__ == "__main__":
    main()