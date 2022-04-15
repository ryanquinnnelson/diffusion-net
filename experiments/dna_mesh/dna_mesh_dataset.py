import os
import sys

import numpy as np

import torch
from torch.utils.data import Dataset

import potpourri3d as pp3d

sys.path.append(os.path.join(os.path.dirname(__file__), "../../src/"))  # add the path to the DiffusionNet src
import diffusion_net


class DNAMeshDataset(Dataset):

    def __init__(self, data_dir, train, k_eig):
        self.train = train  # bool
        self.root_dir = data_dir
        self.k_eig = k_eig
        self.op_cache_dir = os.path.join(data_dir, "cache")

        # store in memory
        self.verts_list = []
        self.faces_list = []

        # Load the meshes & labels
        if self.train:
            with open(os.path.join(self.root_dir, "train.txt")) as f:
                fnames_list = [line.rstrip() for line in f]
        else:
            with open(os.path.join(self.root_dir, "test.txt")) as f:
                fnames_list = [line.rstrip() for line in f]

        print("loading {} files: {}".format(len(fnames_list), fnames_list))

        # Load the actual files
        mesh_path = os.path.join(data_dir, "obj")
        for f in fnames_list:
            fpath = os.path.join(mesh_path, f)

            verts, faces = pp3d.read_mesh(fpath)

            verts = torch.tensor(verts).float()
            faces = torch.tensor(faces)

            # center and unit scale
            verts = diffusion_net.geometry.normalize_positions(verts)

            self.verts_list.append(verts)
            self.faces_list.append(faces)

        # Precompute operators
        output = diffusion_net.geometry.get_all_operators(self.verts_list,
                                                          self.faces_list,
                                                          k_eig=self.k_eig,
                                                          op_cache_dir=self.op_cache_dir)
        # unpack
        self.frames_list, self.massvec_list, self.L_list, self.evals_list, self.evecs_list, self.gradX_list, self.gradY_list = output

    def __len__(self):
        return len(self.verts_list)

    def __getitem__(self, idx):
        result = self.verts_list[idx], self.faces_list[idx], self.frames_list[idx], self.massvec_list[idx], self.L_list[
            idx], self.evals_list[idx], self.evecs_list[idx], self.gradX_list[idx], self.gradY_list[idx]

        return result


print('test')
data_dir = '/Users/ryanqnelson/Downloads'
train = True
k_eig = 128
d1 = DNAMeshDataset(data_dir, train, k_eig)
print('done')
