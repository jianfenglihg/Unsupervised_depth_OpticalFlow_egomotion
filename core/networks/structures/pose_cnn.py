# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn
from attention import CAM_Module


class PoseCNN(nn.Module):
    def __init__(self, num_input_frames):
        super(PoseCNN, self).__init__()

        self.num_input_frames = num_input_frames

        self.convs = {}
        self.convs[0] = nn.Conv2d(3 * num_input_frames, 16, 7, 2, 3)
        self.convs[1] = nn.Conv2d(16, 32, 5, 2, 2)
        self.convs[2] = nn.Conv2d(32, 64, 3, 2, 1)
        self.convs[3] = nn.Conv2d(64, 128, 3, 2, 1)
        self.convs[4] = nn.Conv2d(128, 256, 3, 2, 1)
        self.convs[5] = nn.Conv2d(256, 256, 3, 2, 1)
        self.convs[6] = nn.Conv2d(256, 256, 3, 2, 1)

        self.pose_conv = nn.Conv2d(256, 6 * (num_input_frames - 1), 1)

        self.num_convs = len(self.convs)

        self.relu = nn.ReLU(True)

        self.net = nn.ModuleList(list(self.convs.values()))

        self.query_fc = nn.Linear(14,14)
        self.key_fc   = nn.Linear(14,14)
        self.value_fc = nn.Linear(14,14)

        self.refine_convs = {}
        self.refine_convs[0] = nn.Conv2d(12* (num_input_frames - 1), 6 * (num_input_frames - 1),1,1,0)
        self.refine_convs[1] = nn.Conv2d(6 * (num_input_frames - 1), 6 * (num_input_frames - 1),3,1,1)
        self.refine_convs[2] = nn.Conv2d(6 * (num_input_frames - 1), 6 * (num_input_frames - 1),3,1,1)
        self.refine_convs[3] = nn.Conv2d(6 * (num_input_frames - 1), 6 * (num_input_frames - 1),3,1,1)

        self.refine_net = nn.ModuleList(list(self.refine_convs.values()))

        self.refine_pose_conv = nn.Conv2d(6 * (num_input_frames - 1), 6 * (num_input_frames - 1), 1)


    def atten_refine(self, inputs):
        B,C,H,W = inputs.size()
        inputs = inputs.view([B,C,H*W]) # B C N

        query = self.query_fc(inputs)
        key   = self.key_fc(inputs)
        value = self.value_fc(inputs)

        energy = torch.bmm(query, key.permute([0,2,1]))
        p_mat  = nn.functional.softmax(energy, 1)

        output = torch.bmm(p_mat, value)
        output = torch.cat([inputs, output], 1).view([B,2*C,H,W])

        for i in range(len(self.refine_convs)):
            output = self.refine_convs[i](output)
            output = self.relu(output)

        refine_output = self.refine_pose_conv(output)

        refine_output = refine_output.mean(3).mean(2)
        refine_output = 0.01 * refine_output.view(-1, self.num_input_frames - 1, 6) # B 2 1 6
        
        return refine_output
        



    def forward(self, out):

        for i in range(self.num_convs):
            out = self.convs[i](out)
            out = self.relu(out)
        
        out = self.pose_conv(out)

        delta = self.atten_refine(out)

        out = out.mean(3).mean(2)
        out = 0.01 * out.view(-1, self.num_input_frames - 1, 6) # B 2 1 6

        return out+delta