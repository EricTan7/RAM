"""
    Sinkhorn Algorithm for different settings of Optimal Transport
    Implementation inspired from: https://github.com/PythonOT/POT, thanks
"""


import torch



def Sinkhorn(a, b, M, reg, max_iter=100, thresh=1e-3):
    """
        Sinkhorn Iteration
        Solving Entropic Optimal Transport (EOT)
        Args:
            a: torch.Tensor[B, N], B - batch size, N - number of points in the source distribution
            b: torch.Tensor[B, M], B - batch size, M - number of points in the target distribution
            M: torch.Tensor[B, N, M], cost matrix
            reg: float, regularization strength
            max_iter: int, maximum number of iterations
            thresh: float, convergence threshold
        Returns:
            T: torch.Tensor[B, N, M], transport plan
    """
    K = torch.exp(-M / reg)
    r = torch.ones_like(a)
    c = torch.ones_like(b)
    thresh = 1e-3

    for i in range(max_iter):
        r0 = r
        r = a / torch.matmul(K, c.unsqueeze(-1)).squeeze(-1)
        c = b / torch.matmul(K.permute(0, 2, 1).contiguous(), r.unsqueeze(-1)).squeeze(-1)
        err = (r - r0).abs().mean(dim=1)
        if torch.all(err < thresh):
            break
    T = torch.matmul(r.unsqueeze(-1), c.unsqueeze(-2)) * K
    return T



def Sinkhorn_entropic_unbalanced(a, b, M, reg, reg_m, max_iter=100, thresh=1e-3):
    """
        Sinkhorn Iteration
        Solving Entropic Unbalanced Optimal Transport (EUOT)
        Args:
            a: torch.Tensor[B, N], B - batch size, N - number of points in the source distribution
            b: torch.Tensor[B, M], B - batch size, M - number of points in the target distribution
            M: torch.Tensor[B, N, M], cost matrix
            reg: float, entropy regularization strength
            reg_m: float, marginal regularization strength
            max_iter: int, maximum number of iterations
            thresh: float, convergence threshold
        Returns:
            T: torch.Tensor[B, N, M], transport plan
    """
    if isinstance(reg_m, float) or isinstance(reg_m, int):
        reg_m1, reg_m2 = reg_m, reg_m
    else:
        reg_m1, reg_m2 = reg_m[0], reg_m[1]
    
    u = torch.ones_like(a)
    v = torch.ones_like(b)

    # entropic reg
    K = torch.exp(-M / reg)
    # kl unbalanced
    fi_1 = reg_m1 / (reg_m1 + reg)
    fi_2 = reg_m2 / (reg_m2 + reg)

    thresh = 1e-3
    for i in range(max_iter):
        uprev = u
        vprev = v

        Kv = torch.matmul(K, v.unsqueeze(-1)).squeeze(-1) 
        u = (a / Kv) ** fi_1
        Ktu = torch.matmul(K.permute(0, 2, 1).contiguous(), u.unsqueeze(-1)).squeeze(-1) 
        v = (b / Ktu) ** fi_2

        max_u = torch.cat([torch.max(torch.abs(u), dim=1, keepdim=True)[0], torch.max(torch.abs(uprev), dim=1, keepdim=True)[0], torch.ones((u.shape[0], 1)).cuda()], dim=1)
        max_v = torch.cat([torch.max(torch.abs(v), dim=1, keepdim=True)[0], torch.max(torch.abs(vprev), dim=1, keepdim=True)[0], torch.ones((v.shape[0], 1)).cuda()], dim=1)

        err_u = torch.max(torch.abs(u - uprev), dim=1)[0] / torch.max(max_u, dim=1)[0]
        err_v = torch.max(torch.abs(v - vprev), dim=1)[0] / torch.max(max_v, dim=1)[0]

        err = 0.5 * (err_u.mean() + err_v.mean())
        if err.item() < thresh:
            break

    T = torch.matmul(u.unsqueeze(-1), v.unsqueeze(-2)) * K
    return T



def Sinkhorn_unbalanced(a, b, M, reg_m, div='kl', reg=0, max_iter=100, thresh=1e-3):
    """
        Sinkhorn Iteration
        Solving Unbalanced Optimal Transport (UOT)
        Args:
            a: torch.Tensor[B, N], B - batch size, N - number of points in the source distribution
            b: torch.Tensor[B, M], B - batch size, M - number of points in the target distribution
            M: torch.Tensor[B, N, M], cost matrix
            reg_m: float, marginals regularization strength
            div: regularization method ("kl", "l2")
            max_iter: int, maximum number of iterations
            thresh: float, convergence threshold
        Returns:
            T: torch.Tensor[B, N, M], transport plan
    """
    if isinstance(reg_m, float) or isinstance(reg_m, int):
        reg_m1, reg_m2 = reg_m, reg_m
    else:
        reg_m1, reg_m2 = reg_m[0], reg_m[1]

    G = torch.matmul(a.unsqueeze(-1), b.unsqueeze(-2)) 
    c = torch.matmul(a.unsqueeze(-1), b.unsqueeze(-2))
    assert div in ["kl", "l2"]

    if div == 'kl':
        sum_r = reg + reg_m1 + reg_m2
        r1, r2, r = reg_m1 / sum_r, reg_m2 / sum_r, reg / sum_r
        K = torch.matmul(a.unsqueeze(-1)**r1, b.unsqueeze(-2)**r2) * (c**r) * torch.exp(-M / sum_r)
    elif div == 'l2':
        K = reg_m1 * a.unsqueeze(-1) + reg_m2 * b.unsqueeze(-2) + reg * c - M
        K = torch.max(K, torch.zeros_like(M))
    
    thresh = 1e-3
    for i in range(max_iter):
        Gprev = G

        if div == 'kl':
            Gd = torch.matmul(torch.sum(G, dim=-1, keepdim=True)**r1, torch.sum(G, dim=1, keepdim=True)**r2) + 1e-16
            G = K * G**(r1 + r2) / Gd
        elif div == 'l2':
            Gd = reg_m1 * torch.sum(G, dim=-1, keepdim=True) + \
                reg_m2 * torch.sum(G, dim=1, keepdim=True) + reg * G + 1e-16
            G = K * G / Gd

        err = torch.sqrt(torch.sum((G - Gprev) ** 2, dim=(1,2)).mean())
        if err < thresh:
            break

    return G
    



