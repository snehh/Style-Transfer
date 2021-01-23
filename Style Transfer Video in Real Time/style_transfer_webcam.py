import cv2
import numpy
import torch
import torch.nn as nn
from torchvision import transforms, datasets

class TransformerNetwork(nn.Module):
    def __init__(self):
        super(TransformerNetwork, self).__init__()
        self.ConvBlock = nn.Sequential(
            ConvLayer(3, 32, 9, 1),
            nn.ReLU(),
            ConvLayer(32, 64, 3, 2),
            nn.ReLU(),
            ConvLayer(64, 128, 3, 2),
            nn.ReLU()
        )
        self.ResidualBlock = nn.Sequential(
            ResidualLayer(128, 3),
            ResidualLayer(128, 3),
            ResidualLayer(128, 3),
            ResidualLayer(128, 3),
            ResidualLayer(128, 3)
        )
        self.DeconvBlock = nn.Sequential(
            DeconvLayer(128, 64, 3, 2, 1),
            nn.ReLU(),
            DeconvLayer(64, 32, 3, 2, 1),
            nn.ReLU(),
            ConvLayer(32, 3, 9, 1, norm="None")
        )

    def forward(self, x):
        x = self.ConvBlock(x)
        x = self.ResidualBlock(x)
        out = self.DeconvBlock(x)
        return out
        
class ResidualLayer(nn.Module):
    def __init__(self, channels=128, kernel_size=3):
        super(ResidualLayer, self).__init__()
        self.conv1 = ConvLayer(channels, channels, kernel_size, stride=1)
        self.relu = nn.ReLU()
        self.conv2 = ConvLayer(channels, channels, kernel_size, stride=1)

    def forward(self, x):
        identity = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        out = out + identity
        return out

class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, norm="instance"):
        super(ConvLayer, self).__init__()
        padding_size = kernel_size // 2
        self.reflection_pad = nn.ReflectionPad2d(padding_size)

        self.conv_layer = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

        self.norm_type = norm
        if (norm=="instance"):
            self.norm_layer = nn.InstanceNorm2d(out_channels, affine=True)
        elif (norm=="batch"):
            self.norm_layer = nn.BatchNorm2d(out_channels, affine=True)

    def forward(self, x):
        x = self.reflection_pad(x)
        x = self.conv_layer(x)
        if (self.norm_type=="None"):
            out = x
        else:
            out = self.norm_layer(x)
        return out

class DeconvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, output_padding, norm="instance"):
        super(DeconvLayer, self).__init__()

        padding_size = kernel_size // 2
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding_size, output_padding)

        self.norm_type = norm
        if (norm=="instance"):
            self.norm_layer = nn.InstanceNorm2d(out_channels, affine=True)
        elif (norm=="batch"):
            self.norm_layer = nn.BatchNorm2d(out_channels, affine=True)

    def forward(self, x):
        x = self.conv_transpose(x)
        if (self.norm_type=="None"):
            out = x
        else:
            out = self.norm_layer(x)
        return out

def image_to_tensor(image, max_size=None):
    if (max_size == None):
        image_to_tensor_transform = transforms.Compose([
            # transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.mul(255))
        ])
    else:
        H, W, C = image.shape
        image_size = tuple([int((float(max_size) / max([H, W])) * x) for x in [H, W]])
        image_to_tensor_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.mul(255))
        ])
    tensor = image_to_tensor_transform(image)
    tensor = tensor.unsqueeze(dim=0)
    return tensor

def tensor_to_image(tensor):
    tensor = tensor.squeeze()
    image = tensor.cpu().numpy()
    image = image.transpose(1, 2, 0)
    return image

def style_transfer_color(src, dest):
    src, dest = src.clip(0, 255), dest.clip(0, 255)
    H, W, _ = src.shape
    dest = cv2.resize(dest, dsize=(W, H), interpolation=cv2.INTER_CUBIC)

    dest_gray = cv2.cvtColor(dest, cv2.COLOR_BGR2GRAY) 
    src_yiq = cv2.cvtColor(src, cv2.COLOR_BGR2YCrCb) 
    src_yiq[..., 0] = dest_gray  

    return cv2.cvtColor(src_yiq, cv2.COLOR_YCrCb2BGR).clip(0, 255)  

def style_transfer_webcam(style_transform_path, w=1280, h=720):

    #Device configuration
    device = ("cuda" if torch.cuda.is_available() else "cpu")

    print("Running....")
    transform_net = TransformerNetwork()
    transform_net.load_state_dict(torch.load(style_transform_path))
    transform_net = transform_net.to(device)

    video_cap = cv2.VideoCapture(0)
    video_cap.set(3, w)
    video_cap.set(4, h)


    with torch.no_grad():
        cnt = 0
        while True:
            val, img = video_cap.read()
            img = cv2.flip(img, 1)
            torch.cuda.empty_cache()
            content_tensor = image_to_tensor(img).to(device)
            final_tensor = transform_net(content_tensor)
            final_image = tensor_to_image(final_tensor.detach())
            if (color_preservation):
                final_image = style_transfer_color(img, final_image)
            final_image = final_image / 255
            print(cnt)
            cv2.imshow('Style Transfer Webcam', final_image)
            if cv2.waitKey(1) == 27:
                break
            cnt+=1

    video_cap.release()
    cv2.destroyAllWindows()

style_transform_path = "transforms/mosaic.pth"
color_preservation = False

style_transfer_webcam(style_transform_path, 1280, 720)