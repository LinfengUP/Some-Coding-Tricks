import torch

def my_cross_entropy(input, target, reduction="mean"):
	# input.shape: torch.size([-1, class])
	# target.shape: torch.size([-1])
	# reduction = "mean" or "sum"

    exp = torch.exp(input)

    tmp1 = exp.gather(1, target.unsqueeze(-1)).squeeze()

    tmp2 = exp.sum(1)

    softmax = tmp1 / tmp2

    log = -torch.log(softmax)

    if reduction == "mean": return log.mean()
    else: return log.sum()

