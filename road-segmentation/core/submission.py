import os
import numpy as np
from PIL import Image
from utility import util

class Submission():

    """
    Given a set of predictions, this class will create the following files:
    (if specified in configuration)
    - submission: csv file to upload on kaggle
    - predictions: a 608 x 608 pixel mask where each pixel has a value between 0 and 1
    - overlays: the original image overlayed with the predictions
    - masks: a 608 x 608 pixel mask where each 16x16 patch of pixels is either 0 or 1
    - mask overlays: the original image overlayed with the masks
    """

    def __init__(self, config):
        self.config = config
        self.predictions = {}

    def add(self, prediction, img_id):
        """
        add a prediction of the image with the img_id
        """
        self.predictions[img_id] = prediction
        self.is_built = False

    def write(self):
        """
        first builds the submission and then writes the output files
        """
        print('Starting Submission...')
        self.build()

        if self.config.SUB_WRITE_PREDICTIONS:
            self.write_predictions()

        if self.config.SUB_WRITE_CSV:
            self.write_csv()

        if self.config.SUB_WRITE_MASKS:
            self.write_masks()

        if self.config.SUB_WRITE_OVERLAYS:
            self.write_overlays()

        if self.config.SUB_WRITE_MASK_OVERLAYS:
            self.write_mask_overlays()

        print('Submission Finished')

    def build(self):
        """
        loop over all added predictions and calculate a label per patch
        """
        self.db = {}
        patch_size = 16

        for img_id, prediction in sorted(self.predictions.items()):
            for j in range(0, prediction.shape[1], patch_size):
                for i in range(0, prediction.shape[0], patch_size):
                    patch = prediction[i:i + patch_size, j:j + patch_size]
                    label = self.__patch_to_label(patch)
                    self.db[img_id, j, i] = label

        self.__build_masks()
        self.is_built = True

    def write_predictions(self):
        """
        write predictions to specified directory
        """
        out_dir = self.config.TEST_OUTPUT_DIR + 'predictions/'
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        print(f'Writing Predictions to: {out_dir}')

        for img_id, prediction in self.predictions.items():
            img = util.to_image(prediction)
            img.save(out_dir + f'test_{img_id}.png')

    def write_csv(self):
        """
        write submission csv to specified directory
        """
        assert(self.is_built), 'predictions where added since last build'
        model_name = self.config.MODEL_NAME
        out_dir = self.config.TEST_OUTPUT_DIR + 'csv/'
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        print(f'Writing Csv to: {out_dir}')

        # We start with a new submission file, delete previous one if it exists
        self.submission_file = out_dir + f'submission_{model_name}.csv'
        try:
            os.remove(self.submission_file)
        except OSError:
            pass

        with open(self.submission_file, 'a') as f:
            f.write('id,prediction\n')
            f.writelines('{:03d}_{}_{},{}\n'.format(img_id, j, i, label) for (img_id, j, i), label in sorted(self.db.items()))


    def write_masks(self):
        """
        write masks to specified directory
        """
        assert(self.is_built), 'predictions where added since last build'

        out_dir = self.config.TEST_OUTPUT_DIR + 'masks/'
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        print(f'Writing Masks to: {out_dir}')

        for img_id, mask in self.masks.items():
            img = util.to_image(mask)
            img.save(out_dir + f'test_{img_id}.png')

    def write_mask_overlays(self):
        """
        write mask overlays to specified directory
        """

        assert(self.is_built), 'predictions where added since last build'

        out_dir = self.config.TEST_OUTPUT_DIR + 'mask_overlays/'
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        print(f'Writing Mask Overlays to: {out_dir}')

        for img_id, mask in self.masks.items():
            overlay = np.zeros((mask.shape[0], mask.shape[1], 3))
            overlay[:,:,0] = mask
            overlay = util.to_image(overlay)

            bg = Image.open(self.config.TEST_PATH_TO_DATA + f'/test_{img_id}.png')

            overlay = self.__to_overlay(bg=bg, overlay=overlay, alpha=0.2)
            overlay.save(out_dir + f'test_{img_id}.png')

    def write_overlays(self):
        """
        write overlays to specified directory
        """
        out_dir = self.config.TEST_OUTPUT_DIR + 'overlays/'
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        print(f'Writing Overlays to: {out_dir}')

        for img_id, prediction in self.predictions.items():
            overlay = util.to_image(prediction)
            bg = Image.open(self.config.TEST_PATH_TO_DATA + f'/test_{img_id}.png')

            overlay = self.__to_overlay(bg=bg, overlay=overlay)
            overlay.save(out_dir + f'test_{img_id}.png')

    def __build_masks(self):
        """
        builds the 16x16 patch masks
        """
        patch_size = 16

        patch = {
            0: np.zeros((patch_size,patch_size)),
            1: np.ones((patch_size,patch_size))
        }

        self.masks = {}
        for (img_id, j, i), label in sorted(self.db.items()):
            if img_id not in self.masks:
                self.masks[img_id] = np.zeros((self.config.TEST_IMAGE_SIZE, self.config.TEST_IMAGE_SIZE))
            self.masks[img_id][i:i + patch_size, j:j + patch_size] = patch[label]

    def __to_overlay(self, bg, overlay, alpha=0.5):
        """
        overlays the two images
        """
        bg = bg.convert('RGBA')
        overlay = overlay.convert('RGBA')
        return Image.blend(bg, overlay, alpha)

    def __patch_to_label(self, patch):
        """
        assign a label to the patch
        """
        df = np.mean(patch)
        if df > self.config.TEST_PATCH_FOREGROUND_THRESHOLD:
            return 1
        else:
            return 0
