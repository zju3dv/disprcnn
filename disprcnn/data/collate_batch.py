# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from disprcnn.structures.image_list import to_image_list


class BatchCollator(object):
    """
    From a list of samples from the dataset,
    returns the batched images and targets.
    This should be passed to the DataLoader
    """

    def __init__(self, size_divisible=0):
        self.size_divisible = size_divisible

    def __call__(self, batch):
        transposed_batch = list(zip(*batch))
        images = to_image_list(transposed_batch[0], self.size_divisible)
        targets = transposed_batch[1]
        img_ids = transposed_batch[2]
        return images, targets, img_ids


class DoubleViewBatchCollator(BatchCollator):
    def __call__(self, batch):
        transposed_batch = list(zip(*batch))
        images = {'left': to_image_list([b['left'] for b in transposed_batch[0]], self.size_divisible),
                  'right': to_image_list([b['right'] for b in transposed_batch[0]], self.size_divisible)}
        # images = to_image_list(transposed_batch[0], self.size_divisible)
        # targets = transposed_batch[1]
        targets = {'left': [t['left'] for t in transposed_batch[1]],
                   'right': [t['right'] for t in transposed_batch[1]]}
        img_ids = transposed_batch[2]
        if len(transposed_batch) == 5:
            preds2d = {'left': list(transposed_batch[3]),
                       'right': list(transposed_batch[4])}
            return images, targets, [img_ids, preds2d]
        else:
            return images, targets, img_ids

        # img_ids = {'left': [t['left'] for t in transposed_batch[2]],
        #            'right': [t['right'] for t in transposed_batch[2]]}
