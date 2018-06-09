import os, skimage
import numpy as np
from PIL import Image
from typing import Tuple


def split_into_patches(image: np.ndarray, patch_size: Tuple[int, int, int], stride: int = 1):
    assert(len(patch_size)==3), "patch_size needs 3 elements"

    # check that patch dim and stride allow to evenly divide image into patches
    n_h = (image.shape[0] - patch_size[0] + stride)/ float(stride)
    n_w = (image.shape[1] - patch_size[1] + stride)/ float(stride)

    assert(n_h.is_integer()), "image height, patch height and stride are not matching"
    assert(n_w.is_integer()), "image width, patch width and stride are not matching"

    # return patches
    return skimage.util.view_as_windows(image, patch_size, stride)

def merge_into_image(patches: np.ndarray, image_size: Tuple[int, int, int], stride: int = 1):
    assert(len(image_size)==3), "image_size needs 3 elements"

    rows, cols, _ , patch_h, patch_w, _ = patches.shape

    image = np.zeros(image_size)
    mask = np.zeros(image_size)

    for row in range(rows):
        i = row * stride # calc row idx
        for col in range(cols):
            j = col * stride # calc col idx

            # place patch (row,col) in proper place in image
            image[i : i+patch_h, j : j+patch_w, :] += patches[row, col , 0, :, :, :]
            # note that we added a patch in this place
            mask[i : i+patch_h, j : j+patch_w, :] += 1

    # calc average for overlapping areas
    avg_image = image/mask
    return avg_image.astype('uint8')


def test():

    print("Load Image and Groundtruth")
    image_dir = "./../../data/training/images/"
    gt_dir = "./../../data/training/groundtruth/"

    img = np.asarray(Image.open(image_dir + "satImage_001.png"))
    gt = np.asarray(Image.open(gt_dir + "satImage_001.png"))
    gt = gt.reshape((gt.shape[0], gt.shape[1], 1))

    stride = 60 # controls how much the patches overlap

    print("\nSplit Image and GT into Patches:")
    img_patches = split_into_patches(image=img, patch_size=(100, 100, 3), stride=stride)
    gt_patches = split_into_patches(image=gt, patch_size=(100, 100, 1), stride=stride)

    print('img_patches: ', img_patches.shape)
    print('gt_patches: ', gt_patches.shape)

    print("\nMerge Patches into Image and GT:")
    img_merged = merge_into_image(patches=img_patches, image_size=(img.shape[0], img.shape[1], img.shape[2]), stride=stride)
    gt_merged = merge_into_image(patches=gt_patches, image_size=(gt.shape[0], gt.shape[1], gt.shape[2]), stride=stride)

    print('img_merged: ', img_merged.shape)
    print('gt_merged: ', gt_merged.shape)

    print("\nCheck that Merged Image and GT correspond to original Image and GT:")
    np.testing.assert_array_equal(img,img_merged, "Merged Image not identical to original Image")
    np.testing.assert_array_equal(gt,gt_merged, "Merged GT not identical to original GT")

    print("Success!")

if __name__ == '__main__':
    test()
