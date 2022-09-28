import torch
import numpy as np
import random
import os

from . import utils as utils
from . import decoder as decoder
from .build import DATASET_REGISTRY

import slowfast.utils.logging as logging
from slowfast.utils.env import pathmgr

logger = logging.get_logger(__name__)

class PuzzleHelper():
    """
        V1:
            Inputs: orig images (with indices) -> permute images (with indices)
            => Calculate loss sequentially

        V2: https://github.com/bbrattoli/JigsawPuzzlePytorch/blob/master/select_permutations.py
            Inputs: random permute and orig images with bias_permute (more permute or not)
    """

    def __init__(self, cfg, length=8, jig_classes=1000, bias_whole_image=0.9):
        self.cfg = cfg
        self.jig_classes = jig_classes # permute class = 30
        self.bias_whole_image = bias_whole_image
        self.length = length
        logger.info("Puzzle config: length={}, jig_classes={}, bias={}".format(length, jig_classes, bias_whole_image))
        self._init_permute_data()
        self._create_d_matrix()

        self.version = 'v2.1'
        logger.info("Permute version: {}".format(self.version))

    def _init_permute_data(self):
        filename = 'permutations/permutations_%d_%d.npy' % (self.length, self.jig_classes)
        logger.info("Read file {}...".format(filename))
        all_perm = np.load(filename)
        # from range [1,8] to [0,7]
        if all_perm.min() == 1:
            all_perm = all_perm - 1
        self.permutations = all_perm

    def _create_d_matrix(self):
        self.d_matrix = []
        for perm in self.permutations:
            m = np.ones([self.length, self.length])
            for i in range(self.length):
                for j in range(i+1, self.length):
                    if perm[i] > perm[j]:
                        m[i,j] = -1
                        m[j,i] = -1
            self.d_matrix.append(m)
        self.d_matrix = np.array(self.d_matrix)

    def _get_permute_data(self, frames, label, index):

        if self.version == 'v1':
            perm_frames, perm_indices = decoder.temporal_shuffle(torch.clone(frames))
            indices = torch.arange(0, len(perm_indices))
            frames = utils.pack_pathway_output(self.cfg, frames)
            perm_frames = utils.pack_pathway_output(self.cfg, perm_frames)

            return frames, label, indices, perm_frames, perm_indices, index, {}

        elif self.version == 'v2':
            order = np.random.randint(len(self.permutations) + 1)  # added 1 for class 0: unsorted
            if self.bias_whole_image and self.bias_whole_image > random.random():
                order = 0
            if order == 0:
                indices = torch.arange(0, self.length)
            else:
                indices = torch.as_tensor(self.permutations[order - 1])
                c, t, h, w = frames.shape
                if t > self.length:
                    frames = frames.reshape(c, self.length, t // self.length, h, w)
                frames = frames[:, indices]
                if t > self.length:
                    frames = frames.reshape(c, t, h, w)
            frames = utils.pack_pathway_output(self.cfg, frames)

            return frames, label, indices, index, {}

        elif self.version == 'v2.1':
            order = np.random.randint(len(self.permutations) + 1)  # added 1 for class 0: unsorted
            indices = torch.arange(0, self.length)
            if order == 0:
                perm_indices = torch.arange(0, self.length)
            else:
                perm_indices = torch.as_tensor(self.permutations[order - 1])
            perm_frames = frames.clone()
            c, t, h, w = perm_frames.shape
            if t > self.length:
                perm_frames = perm_frames.reshape(c, self.length, t // self.length, h, w)
            perm_frames = perm_frames[:, perm_indices]
            if t > self.length:
                perm_frames = perm_frames.reshape(c, t, h, w)

            #frames = utils.pack_pathway_output(self.cfg, frames)
            #perm_frames = utils.pack_pathway_output(self.cfg, perm_frames)

            frames = torch.index_select(
                 frames,
                 1,
                 torch.linspace(
                     0, frames.shape[1] - 1, self.cfg.DATA.NUM_FRAMES

                 ).long(),
            )

            perm_frames = torch.index_select(
                 perm_frames,
                 1,
                 torch.linspace(
                     0, perm_frames.shape[1] - 1, self.cfg.DATA.NUM_FRAMES

                 ).long(),
            )


            return frames, label, indices, perm_frames, perm_indices, order, index, {}

        else:
            return frames, label, index, {}


    def _get_d_data(self, frames, label, index):

        if self.version == 'v1':
            order = np.random.randint(len(self.permutations) + 1)  # added 1 for class 0: unsorted
            if order == 0:
                perm_indices = torch.arange(0, self.length)
                perm_mat = torch.ones([self.length, self.length])
            else:
                perm_indices = torch.as_tensor(self.permutations[order - 1])
                perm_mat = torch.as_tensor(self.d_matrix[order - 1])
            perm_frames = frames.clone()
            c, t, h, w = perm_frames.shape
            if t > self.length:
                perm_frames = perm_frames.reshape(c, self.length, t // self.length, h, w)
            perm_frames = perm_frames[:, perm_indices]
            if t > self.length:
                perm_frames = perm_frames.reshape(c, t, h, w)

            perm_frames = torch.index_select(
                 perm_frames,
                 1,
                 torch.linspace(
                     0, perm_frames.shape[1] - 1, self.cfg.DATA.NUM_FRAMES

                 ).long(),
            )

            return perm_frames, label, perm_mat, index, {}

        elif self.version == 'v2.1':
            order = np.random.randint(len(self.permutations) + 1)  # added 1 for class 0: unsorted
            ord_mat = torch.ones([self.length, self.length])
            if order == 0:
                perm_indices = torch.arange(0, self.length)
                perm_mat = torch.ones([self.length, self.length])
            else:
                perm_indices = torch.as_tensor(self.permutations[order - 1])
                perm_mat = torch.as_tensor(self.d_matrix[order - 1])
            perm_frames = frames.clone()
            c, t, h, w = perm_frames.shape
            if t > self.length:
                perm_frames = perm_frames.reshape(c, self.length, t // self.length, h, w)
            perm_frames = perm_frames[:, perm_indices]
            if t > self.length:
                perm_frames = perm_frames.reshape(c, t, h, w)

            frames = torch.index_select(
                 frames,
                 1,
                 torch.linspace(
                     0, frames.shape[1] - 1, self.cfg.DATA.NUM_FRAMES

                 ).long(),
            )

            perm_frames = torch.index_select(
                 perm_frames,
                 1,
                 torch.linspace(
                     0, perm_frames.shape[1] - 1, self.cfg.DATA.NUM_FRAMES

                 ).long(),
            )

            return frames, label, ord_mat, perm_frames, perm_mat, index, {}

        else:
            frames = torch.index_select(
                 frames,
                 1,
                 torch.linspace(
                     0, frames.shape[1] - 1, self.cfg.DATA.NUM_FRAMES

                 ).long(),
            )
            return frames, label, index, {}
