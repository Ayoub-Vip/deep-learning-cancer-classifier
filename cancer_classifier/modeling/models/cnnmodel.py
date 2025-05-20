import torch
from torch import nn

def conv_block(out_channels, kernel_size=3,
                stride=1, padding=1, pool_kernel_size=2,
                pool_stride=2, dropout_prob=0.2):
    return nn.Sequential(
        nn.LazyConv2d(out_channels, kernel_size, stride, padding),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Dropout2d(p=dropout_prob),
        nn.MaxPool2d(pool_kernel_size, pool_stride)
    )

class CNNModel(nn.Module):
    def __init__(self, fc_dropout_prob=0.3, dropout_prob=0.2, num_classes=3):
        super(CNNModel, self).__init__()
    
        self.conv_net = nn.Sequential(
            conv_block(64, 3, dropout_prob=dropout_prob),
            conv_block(128, 5, dropout_prob=dropout_prob),
            conv_block(256, 5, dropout_prob=dropout_prob),
            conv_block(512, 3, 2, dropout_prob=dropout_prob)  
        )
    
        self.fc_layers = nn.Sequential(
            nn.Dropout(fc_dropout_prob),
            nn.LazyLinear(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes)
            )

    def forward(self, x):
        x = self.conv_net(x)

        x = torch.flatten(x, 1)

        x = self.fc_layers(x)

        return x