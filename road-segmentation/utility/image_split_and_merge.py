import os, skimage, itertools
import numpy as np
from PIL import Image
from typing import Tuple
from operator import itemgetter

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


def flatten_patches(patches: np.ndarray, id: str):
    rows, cols, _ , patch_h, patch_w, c = patches.shape
    patches_flatten =  patches.reshape(rows * cols, patch_h, patch_w, c)


    index = [[id,row,col] for row, col in itertools.product(range(rows), range(cols))]
    index = np.array(index)
    assert(patches_flatten.shape[0]==index.shape[0]), "must be"

    return patches_flatten, index

def merge_into_image_from_flatten(patches_flatten: np.ndarray, index: np.ndarray, image_size: Tuple[int, int, int], stride: int = 1):
    assert(len(image_size)==3), "image_size needs 3 elements"

    _, patch_h, patch_w, _ = patches_flatten.shape

    image = np.zeros(image_size)
    mask = np.zeros(image_size)
    check = np.full(image_size, np.nan)
    image_id = int(index[0][0])

    for k, idx in enumerate(index):
        assert(int(idx[0])==image_id), "patches from multiple images"

        row = int(idx[1])
        col = int(idx[2])
        check[row,col] = 1

        patch = patches_flatten[k,:,:,:]

        i = row * stride # calc row idx
        j = col * stride # calc col idx

        # place patch (row,col) in proper place in image
        image[i : i+patch_h, j : j+patch_w, :] += patch

        # note that we added a patch in this place
        mask[i : i+patch_h, j : j+patch_w, :] += 1.0
        check[i : i+patch_h, j : j+patch_w] = 1

    assert(not np.isnan(check).any()), "missing patches"



    # calc average for overlapping areas
    avg_image = image/mask

    print('mask: \n', mask)
    print('image: \n', image)
    print('avg_image: \n', avg_image)

    return avg_image

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

    print("\nFlatten Patches:")
    img_patches_flatten, img_idx = flatten_patches(patches=img_patches, id=7)
    gt_patches_flatten, gt_idx = flatten_patches(patches=gt_patches, id=8)

    print('img_patches_flatten: ', img_patches_flatten.shape)
    print('gt_patches_flatten: ', gt_patches_flatten.shape)

    # del img_idx[9:11] # check that missing patches detection works
    # img_idx[5][0] = 11 # check that double image detection works

    print("\nMerge Patches into Image and GT:")
    img_merged = merge_into_image(patches=img_patches, image_size=(img.shape[0], img.shape[1], img.shape[2]), stride=stride)
    gt_merged = merge_into_image(patches=gt_patches, image_size=(gt.shape[0], gt.shape[1], gt.shape[2]), stride=stride)

    print('img_merged: ', img_merged.shape)
    print('gt_merged: ', gt_merged.shape)


    print("\nMerge Patches into Image and GT from Flatten:")
    img_merged_flatten = merge_into_image_from_flatten(patches_flatten=img_patches_flatten, index=img_idx, image_size=(img.shape[0], img.shape[1], img.shape[2]), stride=stride)
    gt_merged_flatten = merge_into_image_from_flatten(patches_flatten=gt_patches_flatten, index=gt_idx, image_size=(gt.shape[0], gt.shape[1], gt.shape[2]), stride=stride)

    print('img_merged_flatten: ', img_merged_flatten.shape)
    print('gt_merged_flatten: ', gt_merged_flatten.shape)


    print("\nCheck that Merged Image and GT correspond to original Image and GT:")
    np.testing.assert_array_equal(img,img_merged, "Merged Image not identical to original Image")
    np.testing.assert_array_equal(gt,gt_merged, "Merged GT not identical to original GT")

    np.testing.assert_array_equal(img,img_merged_flatten, "Merged Image from Flatten not identical to original Image")
    np.testing.assert_array_equal(gt,gt_merged_flatten, "Merged GT from Flatten not identical to original GT")

    print("Success!")

if __name__ == '__main__':
    test()
