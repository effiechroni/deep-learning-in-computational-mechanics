import torch
from torch.utils.data import Dataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def Normalize(data, dim=None):
    if dim == None:
        norm = [torch.min(data), torch.max(data)]
    else:
        norm = [torch.min(data, dim)[0], torch.max(data, dim)[0]]
    return (data - norm[0]) / (norm[1] - norm[0]) * 2 - 1, norm


def Denormalize(data, norm):
    return 0.5 * (data + 1) * (norm[1] - norm[0]) + norm[0]


class FullWaveFormInversionDataset1D(Dataset):
    def __init__(self, settings, device):
        self.device = device
        self.len = settings.numberOfSamples[0]
        N = settings.N[0]
        Nx = settings.Nx[0]

        # load to cpu memory
        self.U = torch.zeros((self.len, 2, 2, N + 1))
        self.C = torch.zeros((self.len, Nx + 3))
        self.Coeff = torch.zeros((self.len, 3))

        for i in range(self.len):
            self.U[i] = torch.load("dataset1DFWI/measurement" + str(i) + ".pt", weights_only=True)
            self.C[i] = torch.load("dataset1DFWI/material" + str(i) + ".pt", weights_only=True)
            self.Coeff[i] = torch.load("dataset1DFWI/materialCoefficients" + str(i) + ".pt", weights_only=True)

        # normalize data 
        self.U, self.Unorm = Normalize(self.U)
        self.C, self.Cnorm = Normalize(self.C)
        self.Coeff, self.Coeffnorm = Normalize(self.Coeff, dim=0)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        # return on gpu memory if available
        u = self.U[idx].to(device)
        c = self.C[idx].to(device)
        coeff = self.Coeff[idx].to(device)
        return u, c, coeff
