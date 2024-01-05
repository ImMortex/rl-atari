import torch


def check_cuda() -> bool:
    is_cuda: bool = torch.cuda.is_available()
    print(torch.cuda.is_available())
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    if is_cuda:
        version = torch.version.cuda
        print("cuda " + version)
    return is_cuda

if __name__ == '__main__':
    check_cuda()
