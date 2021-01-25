# Import libraries
import argparse
import time
import os, os.path 				
import sys
import numpy as np	
import cv2					
import torch 	
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch import optim					
from torchvision import transforms, models	
from torchvision.utils import save_image		
from PIL import Image			
import matplotlib.pyplot as plt 

# Count of total number of frames of input video
frames_count = 0				

# input_content_frames_dir, output_styletransfer_frames_dir, output_video_dir will be created during the execution of the python code
input_content_frames_dir = "input_content_frames"
input_style_frames_dir = "input_style_frames"
output_styletransfer_frames_dir = "output_styletransfer_frames"
output_video_dir = "output_video"

def load_image(image_path, max_size=400, shape=None):
    image = Image.open(image_path).convert('RGB')
    if max(image.size) > max_size:
        image_size = max_size
    else:
        image_size = max(image.size)
    if shape is not None:
        image_size = shape
    image_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                            (0.229, 0.224, 0.225))])
    image = image_transform(image)[:3, :, :].unsqueeze(0)
    return image
    
    
def video_to_image(video_directory):
    try:
        os.mkdir(input_content_frames_dir)
    except OSError:
        print("Directory creation failed: %s" %input_content_frames_dir)
    else:
        print("Directory creation successful: %s" %input_content_frames_dir)
    capture_video = cv2.VideoCapture(video_directory)
    success, frame = capture_video.read()
    count = 1
    while success:
        cv2.imwrite(input_content_frames_dir + '/frame_%d.jpg' % count, frame)
        success, frame = capture_video.read()
        print("Successfully read frame: ", count)
        count += 1
    global frames_count
    frames_count = count

def get_features(image, model, layers=None):
    if layers is None:
        layers = {'0': 'conv1_1',
                  '5': 'conv2_1',
                  '10': 'conv3_1',
                  '19': 'conv4_1',
                  '21': 'conv4_2',		# Content layer
                  '28': 'conv5_1',}
    features = {}
    x = image
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[layers[name]] = x
    return features

def gram_matrix(tensor):
    batch_size, depth, height, width = tensor.size()
    tensor = tensor.view(depth, height * width)
    tensor_t = tensor.t()
    gram = torch.mm(tensor, tensor_t)
    return gram
    
def image_convert(tensor):
    image = tensor.to("cpu").clone().detach()
    image = image.numpy().squeeze()
    image = image.transpose(1, 2, 0)
    image = image * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))
    image = image.clip(0, 1)
    return image


def style_transfer_process(content_image_dir, style_image_dir):
    try:
        os.mkdir(output_styletransfer_frames_dir)
    except OSError:
        print("Directory creation failed: %s" %output_styletransfer_frames_dir)
    else:
        print("Directory creation successful: %s" %output_styletransfer_frames_dir)
    num_content_images = len([name for name in os.listdir(content_image_dir) if os.path.isfile(os.path.join(content_image_dir, name))])
    num_style_images = len([name for name in os.listdir(style_image_dir) if os.path.isfile(os.path.join(style_image_dir, name))])
    #print("Num_content_img: {}, Num_style_imgs: {}".format(num_content_images, num_style_images))
    # For multiple style frames switch
    #frames_with_current_style = num_content_images // (num_style_images)
    c_count = 1
    s_count = 1
    while(c_count <= num_content_images):
        content_image = load_image(content_image_dir + '/frame_' + str(c_count) + '.jpg').to(device)
        #if(c_count % frames_with_current_style == 0):
        #	s_count += 1
        style_image = load_image(style_image_dir + '/style_' + str(s_count) + '.jpg', shape=content_image.shape[-2:]).to(device)
        content_features = get_features(content_image, Model)
        style_features = get_features(style_image, Model)
        style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_features}
        final_image = content_image.clone().requires_grad_(True).to(device)
        style_weights = {'conv1_1': 1.,
                         'conv2_1': 0.8,
                         'conv3_1': 0.2,
                         'conv4_1': 0.2,
                         'conv5_1': 0.2}
        #hyperparameters
        alpha = 1
        beta = 1e8
        epochs = 400
        optimizer = optim.Adam([final_image], lr=0.003)
        for i in range(1, epochs+1):
            final_image_features = get_features(final_image, Model)
            content_loss = torch.mean((final_image_features['conv4_2'] - content_features['conv4_2']) ** 2)
            style_loss = 0
            for layer in style_weights:
                target_feature = final_image_features[layer]
                batch_size, depth, height, width = target_feature.shape
                target_gram = gram_matrix(target_feature)
                style_gram = style_grams[layer]
                layer_style_loss = style_weights[layer] * torch.mean((target_gram - style_gram) ** 2)
                style_loss += layer_style_loss / (depth * height * width)
            total_loss = alpha*content_loss + beta*style_loss
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
        final_image = image_convert(final_image)
        plt.imsave(output_styletransfer_frames_dir + '/st_frame_%d.jpg' % c_count, final_image)
        print("Style Tranfer completed on image: ", c_count)
        c_count += 1
        if(c_count <= frames_count):
            continue

def image_to_video(st_output_dir):
    try:
        os.mkdir(output_video_dir)
    except OSError:
        print("Directory creation failed: %s" %output_video_dir)
    else:
        print("Directory creation successful: %s" %output_video_dir)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    img = cv2.imread(st_output_dir + '/st_frame_1.jpg')
    height, width, layers = img.shape
    video = cv2.VideoWriter(output_video_dir + '/style_transfered_video.avi', fourcc, 25, (width, height))

    for i in range(1, frames_count):
        video.write(cv2.imread(st_output_dir + '/st_frame_' + str(i) + '.jpg'))
    cv2.destroyAllWindows()
    video.release()

# Load pre-trained model and freezing the weights:
Model = models.vgg19(pretrained=True).features
for x in Model.parameters():
    x.requires_grad_(False)
    
# Device configuration
if torch.cuda.is_available():
    print("Running on a GPU")
else:
    print("Running on a CPU")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
Model.to(device)

#MAIN
# Converting input video to content image frames
video_to_image(sys.argv[1])

# Applying Style transfer process on the frames
style_transfer_process(input_content_frames_dir, input_style_frames_dir)

# Convert style transfer processed image frames to a video
image_to_video(output_styletransfer_frames_dir)