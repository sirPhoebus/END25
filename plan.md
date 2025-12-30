let's start by implementing a baseline TRM in PyTorch. We'll build it modular so we can easily experiment with your proposed merge: adding a lightweight hierarchical element (e.g., alternating recursion frequencies or dual z states within the same network).
The official TRM repo uses a 2-layer Transformer with attention for ARC (MLP variant for Sudoku), but to keep it runnable immediately and lightweight, I'll start with a configurable setup:

Grid embedding (colors 0-9 → learned embeddings + positional via RoPE).
Single shared network (Transformer block).
Inner recursion: update z multiple times (n=6 like the paper).
Outer: update y once per supervision step.
Deep supervision across steps.
Simple halting head (sigmoid probability to stop early, but fixed max steps for now).

We'll need the ARC dataset handling too. The public training set is at https://github.com/fchollet/ARC-AGI/tree/master/data/training.
First, let's create a clean project structure.
requirements.txt
texttorch>=2.0.0
numpy
einops  # for easier reshaping
rotary-embedding-torch  # for RoPE
tqdm
README.md (initial)
text# Optimized TRM for ARC-AGI-2

Baseline implementation of Tiny Recursive Model (TRM) from "Less is More: Recursive Reasoning with Tiny Networks" (2025).

Goal: Reproduce ~45% on ARC-AGI-1 public eval, then merge hierarchical ideas for better ARC-AGI-2 generalization.

Run:
- Download training tasks from https://lab42global.github.io/arc-dataset-collector/
- python train.py
Now, the core model file.
model.py
Pythonimport torch
import torch.nn as nn
import torch.nn.functional as F
from rotary_embedding_torch import RotaryEmbedding
from einops import rearrange
from tqdm import tqdm
import json
import os
import numpy as np

class TRM(nn.Module):
    def __init__(self, grid_size=30, colors=10, dim=512, heads=8, layers=2, n_inner=6, max_steps=16):
        super().__init__()
        self.grid_size = grid_size
        self.colors = colors
        self.dim = dim
        self.n_inner = n_inner  # inner latent updates per outer step
        self.max_steps = max_steps

        # Embeddings
        self.color_emb = nn.Embedding(colors, dim)
        self.rotary = RotaryEmbedding(dim // heads)

        # Shared transformer block
        def block():
            return nn.TransformerEncoderLayer(
                d_model=dim,
                nhead=heads,
                dim_feedforward=dim * 4,
                activation="gelu",
                batch_first=True,
                norm_first=True
            )
        self.net = nn.Sequential(*[block() for _ in range(layers)])

        # Heads
        self.y_head = nn.Linear(dim, colors)  # predict grid tokens
        self.halt_head = nn.Linear(dim, 1)     # halting probability

    def forward(self, x_grids, y_init=None, steps=None, train=False):
        """
        x_grids: (b, demos, h, w) - input demos + test input concatenated
        y_init: initial guess for test output grid, or random if None
        steps: number of refinement steps (default max_steps)
        train: if True, return losses for deep supervision
        """
        b, demos, h, w = x_grids.shape
        assert h == w == self.grid_size

        steps = steps or self.max_steps

        # Embed x (fixed)
        x_flat = rearrange(x_grids, 'b d h w -> b (d h w)')
        x_emb = self.color_emb(x_flat.long())  # (b, seq, dim)
        x_emb = self.rotary.rotate_queries_or_keys(x_emb)

        # Initial y (random or provided)
        if y_init is None:
            y_init = torch.randint(0, self.colors, (b, h, w), device=x_grids.device)
        y = rearrange(y_init, 'b h w -> b (h w)')
        y_emb = self.color_emb(y.long())

        losses = []
        halts = []
        ys = []

        z = torch.zeros(b, h*w, self.dim, device=x_grids.device)  # initial latent

        for step in range(steps):
            for inner in range(self.n_inner):
                # Concat x + y + z embeddings along sequence
                inp = torch.cat([x_emb, y_emb, z], dim=1)  # (b, 3*seq, dim)
                inp = self.rotary.rotate_queries_or_keys(inp)

                out = self.net(inp)  # (b, 3*seq, dim)
                z = out[:, - (h*w):, :]  # update z from last segment

            # Update y from z (and current y?)
            inp_y = torch.cat([x_emb, y_emb, z], dim=1)
            inp_y = self.rotary.rotate_queries_or_keys(inp_y)
            out_y = self.net(inp_y)
            y_new = self.y_head(out_y[:, - (h*w):, :])  # logits over colors
            y_emb = self.color_emb(y_new.argmax(-1))  # discrete for next

            halt_prob = self.halt_head(z.mean(1)).sigmoid()

            ys.append(y_new)
            halts.append(halt_prob)

            if train:
                # Deep supervision: loss on this step's y if we have target
                # We'll handle target in training loop
                pass

        if train:
            return ys, losses  # losses filled in trainer
        else:
            return ys[-1].argmax(-1).view(b, h, w), halts
This is a simplified but faithful starting point—separate y (solution) and z (reasoning), inner recursion on z, outer refinement of y, shared net.
Next step: data loader for ARC tasks (each task has multiple demo pairs + test input, predict test output).
data.py
Pythonimport json
import numpy as np
from torch.utils.data import Dataset

class ARCDataset(Dataset):
    def __init__(self, path, grid_size=30):
        self.tasks = []
        for file in os.listdir(path):
            if file.endswith('.json'):
                with open(os.path.join(path, file)) as f:
                    task = json.load(f)
                    self.tasks.append(task)
        self.grid_size = grid_size

    def __len__(self):
        return len(self.tasks)

    def __getitem__(self, idx):
        task = self.tasks[idx]
        # For simplicity: concatenate all train input+output as "x", test input as part of x, target = test output
        # Better: multiple demos, but pad to max demos
        train_inputs = [np.array(ex['input']) for ex in task['train']]
        train_outputs = [np.array(ex['output']) for ex in task['train']]
        test_input = np.array(task['test'][0]['input'])

        # Pad grids to grid_size
        def pad(g):
            p = np.zeros((self.grid_size, self.grid_size), dtype=int)
            p[:g.shape[0], :g.shape[1]] = g
            return p

        x_grids = [pad(np.array(i)) for i in train_inputs + [test_input]]
        x_grids = np.stack(x_grids)  # (demos+1, h, w)

        target = pad(np.array(task['test'][0]['output']))

        return torch.from_numpy(x_grids), torch.from_numpy(target)
We can refine this (e.g., separate demos from test input, mask output in x).