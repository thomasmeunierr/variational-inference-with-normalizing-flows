import torch


def p_z(z):

    z1, z2 = torch.chunk(z, chunks=2, dim=1)
    norm = torch.sqrt(z1 ** 2 + z2 ** 2)

    exp1 = torch.exp(-0.5 * ((z1 - 2) / 0.8) ** 2)
    exp2 = torch.exp(-0.5 * ((z1 + 2) / 0.8) ** 2)
    u = 0.5 * ((norm - 4) / 0.4) ** 2 - torch.log(exp1 + exp2)

    return torch.exp(-u)

def p_zz(z) :

    z1, z2 = torch.chunk(z, chunks=2, dim=1)
    w1, w2 = torch.sin(2 * torch.pi * z1 / 4), 3 * torch.exp( - ((z2 - 1)/0.6)**2 / 2) 

    e1, e2 =   torch.exp( - ((z2 - w1)/0.35)**2 / 2) , torch.exp( - ((z2 - w1 + w2)/0.35)**2 / 2)

    return (e1 + e2)

def sigma(x):

    return 1 / (1 + torch.exp(-x))

def p_zzz(z) :

    z1, z2 = torch.chunk(z, chunks=2, dim=1)
    w1, w3 = torch.sin(2 * torch.pi * z1 / 4), 3*sigma((z1 -1)/3)

    e1, e2 =   torch.exp( - ((z2 - w1)/0.4)**2 / 2) , torch.exp( - ((z2 - w1 + w3)/0.35)**2 / 2)

    return (e1 + e2)
