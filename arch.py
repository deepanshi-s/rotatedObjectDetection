import torch
import torch.nn as nn
import torch.nn.functional as F

class convolutionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2, padding=1):
        """ResNet convolution block

        Args:
            in_channels (int): number of channels of the input features
            out_channels (int): number of channels of the output features
            stride (int): stride for the convolution operation, Defaults to 2.
            padding (int): padding for the convolution operation. Defaults to 1.
        """
        super(convolutionBlock, self).__init__()
        self.conv1 = nn.Conv2d(
                in_channels, 
                out_channels, 
                kernel_size=3,
                stride=stride,
                padding=padding
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU()
        self.conv2 = nn.Conv2d(
                out_channels, 
                out_channels, 
                kernel_size=3,
                padding=padding
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2),
                nn.BatchNorm2d(out_channels)
            )   
        
    def forward(self, x):
        skip = x
        x = self.bn1(self.conv1(x))
        x = self.activation(x)
        x = self.bn2(self.conv2(x))
        
        skip_connect = self.skip(skip)
        
        x = x + skip_connect
        x = self.activation(x)
        return x
        

class FPN(nn.Module):
    """FPN model with ResNet18 encoder block
       Classification operation takes in the features from the bottleneck
       Regression is based on the bottom of the pyramid

    Add paper reference:
    
    """
    def __init__(self):
        super(FPN, self).__init__()
        filters = [16, 32, 64, 128, 256]
        
        self.conv1 = nn.Sequential(
                nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(8),
                nn.MaxPool2d(2),
                nn.Conv2d(8, filters[0], kernel_size=3, padding=1, stride=1),
                nn.BatchNorm2d(filters[0]),
                nn.MaxPool2d(2)
        )
        
        #Resnet type encoder
        self.conv2 = convolutionBlock(filters[0], filters[1], stride=2, padding=1)
        self.conv3 = convolutionBlock(filters[1], filters[2], stride=2, padding=1)
        self.conv4 = convolutionBlock(filters[2], filters[3], stride=2, padding=1)
        self.conv5 = convolutionBlock(filters[3], filters[4], stride=2, padding=1)
        
        downsampleChannel = 256
        
        self.bottleneck = nn.Conv2d(filters[4], downsampleChannel, 1)
        
        #Parallel connections
        self.conv2Parallel = nn.Conv2d(filters[1], downsampleChannel, kernel_size=1)
        self.conv3Parallel = nn.Conv2d(filters[2], downsampleChannel, kernel_size=1)
        self.conv4Parallel = nn.Conv2d(filters[3], downsampleChannel, kernel_size=1)
        
        self.convHead = nn.Conv2d(downsampleChannel, downsampleChannel, kernel_size=3, padding=1)
        
        self.pooling = nn.AdaptiveAvgPool2d((1, 1))
        
        self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(downsampleChannel, 1)
        )
        
        self.regression = nn.Sequential(
                nn.Flatten(),
                nn.Linear(downsampleChannel, 5)
        )
        self.sigmoid = nn.Sigmoid()
        
    
    def forward(self, x):
        c1 = self.conv1(x)
        c2 = self.conv2(c1)
        c3 = self.conv3(c2)
        c4 = self.conv4(c3)
        c5 = self.conv5(c4)
        
        p5 = self.bottleneck(c5)
        p4 = F.interpolate(p5, size=c4.shape[-2:], mode="nearest") + self.conv4Parallel(c4)
        p3 = F.interpolate(p4, size=c3.shape[-2:], mode="nearest") + self.conv3Parallel(c3)
        p2 = F.interpolate(p3, size=c2.shape[-2:], mode="nearest") + self.conv2Parallel(c2)
        
        classifierFeatures = self.pooling(p5)
        regressionFeatures = self.pooling(p5)
        
        classification = self.classifier(classifierFeatures)
        classification = self.sigmoid(classification)
        regression = self.regression(regressionFeatures)
        regression = self.sigmoid(regression)
        
        starProb = classification.view(x.shape[0], 1)
        bbox = regression.view(x.shape[0], 5)
        
        
        return torch.cat((starProb, bbox), dim=1)
        
# Run file to see summary
if __name__ == "__main__":
    from torchsummary import summary
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inp = torch.rand((2, 1, 200, 200))
    net = FPN()
    net.to(device)
    inp = inp.to(device)
    out = net(inp)

    # print(out.shape)
    summary(net, inp.shape[1:])
    # print(net)
