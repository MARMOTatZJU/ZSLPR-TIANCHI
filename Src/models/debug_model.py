import torch
from ResNeXt_DenseNet import resnet50

if __name__ == '__main__':
    inputs = torch.randn((5, 3, 80, 80))
    model = resnet50(300)
    outputs = model(inputs)
    print(outputs.shape)