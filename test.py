import torch

if __name__ == "__main__":
    x = torch.tensor([2.])
    x.requires_grad_(True)
    y = torch.tensor([2.])
    a = x + y
    b = a + y
    c = a + b
    print(c.requires_grad)
    # out.backward()