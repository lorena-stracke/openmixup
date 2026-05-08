import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from .builder import MODELS
import os
import matplotlib.image as mpimage
import torchvision.utils as vutils
import matplotlib.colors as c
from torch import tensor
import torch.nn.functional as F

# helper methods for preprocessing
def calculate_weight(channels, depth, single_color, color_opponency, luminance_green_bias):
    weight_array = np.ones((channels, depth * 3, 1, 1))

    if depth == 1:
        # preprocessing with color mapping but wihtout contrast 
        if not single_color and not color_opponency and not luminance_green_bias: # Luminance mapping without contrast
            weight_array[:, :3, :, :] *= 1 / 3
        elif color_opponency and not single_color and not luminance_green_bias: # Color-opponency mapping without contrast
            weight_array[0,:,:,:] = 1/3
            for i in range(3):
                weight_array[1, i, 0, 0] = [0.5, -0.5, 0][i]
                weight_array[2, i, 0, 0] = [-0.5 / 3, -0.5 / 3, 1 / 3][i]
        elif luminance_green_bias and not color_opponency and not single_color: # Luminance_green_bias mapping without contrast
            weight_array[:, 0, :, :] *= 0.299
            weight_array[:, 1, :, :] *= 0.587
            weight_array[:, 2, :, :] *= 0.114
    else:
        #preprocessing with color mapping and contrast (default luminance mapping)
        weight_array[:, :3, :, :] *= 1 / 3
        weight_array[:, 3:, :, :] *= -(1 / (depth * 3 - 3))

        if channels == 3 and single_color: # single-color mapping with contrast
            for c in range(channels):
                weight_array[c, :, 0, 0] = 0  
                weight_array[c, c, 0, 0] = 1  

                for i in range(1, depth):
                    weight_array[c, i * 3 + c, 0, 0] = -1 / (depth - 1)

        elif channels == 3 and color_opponency: # color-opponency mapping with contrast
            copy = weight_array[0, :, :, :]
            weight_array = np.zeros((channels, depth * 3, 1, 1))
            weight_array[0, :, :, :] = copy
            base_r_g = [0.5, -0.5, 0]
            r_g_value = [-(1 / ((depth - 1) * 2)), (1 / ((depth - 1) * 2)), 0]
            base_b_y = [-0.5 / 3, -0.5 / 3, 1 / 3]
            b_y_value = [(0.5 / (depth * 3 - 3)), (0.5 / (depth * 3 - 3)), -(1 / (depth * 3 - 3))]

            for i in range(depth * 3):
                if i < 3:
                    weight_array[1, i, 0, 0] = base_r_g[i]
                    weight_array[2, i, 0, 0] = base_b_y[i]
                else:
                    index = (i - 3) % 3
                    weight_array[1, i, 0, 0] = r_g_value[index]
                    weight_array[2, i, 0, 0] = b_y_value[index]

    return weight_array

def create_blur_kernel():
    kernel = np.zeros((3, 3))

    for i in range(3):
        for j in range(3):
            kernel[i, j] = 1 / 9

    return kernel

def save_image(image_tensor, where, save_name, channels, path, training):


    if training:
        savepath = path + "/images"
    else:
        savepath = path + "/images_test"


    os.makedirs(savepath, exist_ok=True)

    image_name = save_name + "_" + where
    # Path(path + "/" + savepath +"/" + save_name).mkdir(parents=True, exist_ok=True)

    cmap_R = c.LinearSegmentedColormap.from_list("cmap_R",
                                                 ['black', 'white', '#f00'])  # hier sind die colormaps definiert,
    cmap_G = c.LinearSegmentedColormap.from_list("cmap_G", ['black', 'white',
                                                            '#0f0'])  # in denen die channel angezeigt werden
    cmap_B = c.LinearSegmentedColormap.from_list("cmap_B", ['black', 'white', '#00f'])

    boundary_red = max(torch.max(image_tensor[0]), -torch.min(image_tensor[0]))

    f, axarr = plt.subplots(5, 4, figsize=(20, 20))

    img_norm = (image_tensor - torch.min(image_tensor)) / (torch.max(image_tensor) - torch.min(image_tensor))
    # img_norm = image_tensor / torch.max(image_tensor)
    axarr[0][0].imshow(img_norm.detach().cpu().permute(1, 2, 0))
    axarr[0][1].imshow(img_norm[0].detach().cpu(), cmap=cmap_R)
    vutils.save_image(img_norm, "" + savepath + "/" + image_name + "RGB.png")
    mpimage.imsave("" + savepath + "/" + image_name + "-r.png", img_norm[0].detach().cpu(), cmap=cmap_R,
                   vmin=-boundary_red, vmax=boundary_red)

    if channels > 1:
        boundary_green = max(torch.max(image_tensor[1]), -torch.min(image_tensor[1]))
        boundary_blue = max(torch.max(image_tensor[2]), -torch.min(image_tensor[2]))
        axarr[0][2].imshow(img_norm[1].detach().cpu(), cmap=cmap_G)
        axarr[0][3].imshow(img_norm[2].detach().cpu(), cmap=cmap_B)
        mpimage.imsave("" + savepath + "/" + image_name + "-g.png", img_norm[1].detach().cpu(), cmap=cmap_G,
                       vmin=-boundary_green, vmax=boundary_green)
        mpimage.imsave("" + savepath + "/" + image_name + "-b.png", img_norm[2].detach().cpu(), cmap=cmap_B,
                       vmin=-boundary_blue, vmax=boundary_blue)

    plt.close()

@MODELS.register_module()
class BlurPreprocessing(nn.Module):
    def __init__(self, blur_bool, blur_depth, single_color, color_opponency, channels, path, training, black_white, normalize, sparsity_threshold, sparsity_type, change_range, sparse_baseline, use_reflect_padding_for_blurring):
        super().__init__()
        self.blur = blur_bool
        self.num_images = blur_depth + 1
        self.single_color = single_color
        self.color_opponency = color_opponency
        self.channels = channels
        self.write = True
        self.path = path
        self.training = training
        self.black_white = black_white
        self.normalize = normalize
        self.sparsity_threshold = sparsity_threshold
        self.sparsity_type = sparsity_type
        self.change_range = change_range
        self.sparse_baseline = sparse_baseline 
        self.use_reflect_padding_for_blurring = use_reflect_padding_for_blurring

        if self.blur:

            blur_kernel = create_blur_kernel()
            self.blur_kernel = tensor(np.array([[blur_kernel],
                                                                  [blur_kernel],
                                                                  [blur_kernel]]), requires_grad=False).float()


            weight_array = calculate_weight(self.channels, self.num_images, self.single_color, self.color_opponency, self.black_white)
            self.weight_array = tensor(weight_array, requires_grad=False).to(torch.get_default_dtype())

            print("preprocessing")
            print(self.blur_kernel)
            print(self.weight_array)
            print(f"blur: {self.blur}, blur_depth: {blur_depth}, single_color: {single_color}, color_opponency: {color_opponency}, channels: {channels}, black_white: {black_white}, normalize: {normalize}, sparsity_threshold: {sparsity_threshold}, sparsity_type: {sparsity_type}, sparse_baseline: {sparse_baseline}, use_reflect_padding_for_blurring: {self.use_reflect_padding_for_blurring}")

            if self.normalize:
                print("Normalizing the images")
            if self.sparsity_threshold > 0.0:
                if self.sparsity_type == 'percentage':
                    print(f"Creating sparsity based on percentage: {self.sparsity_threshold}")
                else:
                    print(f"Creating sparsity with threshold {self.sparsity_threshold}")



    def forward(self, x):
        if self.blur and not self.sparse_baseline:

            if self.write:
                print("saving image before preprocessing")
                save_image(x[0], "before", "image", self.channels, self.path, self.training)

            concat_image = x

            # import ipdb; ipdb.set_trace()
            for i in range(self.num_images - 1):
                x = F.pad(x, (1, 1, 1, 1), mode="reflect")
                x = F.conv2d(x, self.blur_kernel.to(x.device), stride=(1, 1), padding=0, groups=3)
                concat_image = torch.concat([concat_image, x], dim=1)

            x = F.conv2d(concat_image, self.weight_array.to(x.device))          

        if self.sparsity_threshold > 0.0:

            #percentage based sparsity
            if self.sparsity_type == 'percentage':
                num_elements = x.numel()
                k = int(self.sparsity_threshold * num_elements)

                if k > 0:
                    abs_vals = x.abs().flatten()
                    threshold = torch.topk(abs_vals, k, largest=False).values.max()
                    sparse_image = torch.where(x.abs() <= threshold, torch.tensor(0.0, device=x.device), x)
                    x = sparse_image
            else:
                #value based sparsity
                sparse_image = torch.where(x.abs() < self.sparsity_threshold, torch.tensor(0.0, device=x.device), x)
                x = sparse_image
            
                if not self.training:
                    image_pixel_number = x.numel()
                    number_of_zero_pixels = (sparse_image == 0.0).sum().item()
                    print(f"Eval: Percentage of zero pixels in the sparse image: {number_of_zero_pixels/image_pixel_number}")

        if self.write:
            image_pixel_number = x.numel()
            if not self.training:
                number_of_zero_pixels = (x == 0.0).sum().item()
                print(f"Percentage of zero pixels in the sparse image: {number_of_zero_pixels/image_pixel_number}")
            print("saving image after preprocessing")
            save_image(x[0].abs(), "after", "image", self.channels, self.path, self.training)

            self.write = False

        return x


