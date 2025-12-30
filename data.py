import json
import os
import random
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

class ARCDataset(Dataset):
    def __init__(self, data_path, mode='train', augment=False, grid_size=30):
        """
        Args:
            data_path: Path to directory.
            mode: 'train' or 'test'.
            augment: Apply augmentations (usually only for mode='train').
        """
        self.tasks = []
        self.mode = mode
        self.augment = augment
        self.grid_size = grid_size
        
        if os.path.exists(data_path):
            for file in os.listdir(data_path):
                if file.endswith('.json'):
                    try:
                        with open(os.path.join(data_path, file), 'r') as f:
                            task = json.load(f)
                            self.tasks.append(task)
                    except Exception as e:
                        print(f"Error loading {file}: {e}")
        else:
            print(f"Warning: Data path {data_path} does not exist.")

    def __len__(self):
        return len(self.tasks)

    def pad_grid(self, grid):
        """Pads a grid to (grid_size, grid_size). Returns mask as well."""
        grid = np.array(grid)
        h, w = grid.shape
        padded = np.zeros((self.grid_size, self.grid_size), dtype=int)
        h = min(h, self.grid_size)
        w = min(w, self.grid_size)
        padded[:h, :w] = grid[:h, :w]
        
        mask = np.zeros((self.grid_size, self.grid_size), dtype=float)
        mask[:h, :w] = 1.0
        return padded, mask

    def apply_augmentation(self, task):
        # ... (keep existing implementation, assuming it's available in context effectively or I just don't touch it if I don't select it)
        # Wait, I am executing 'Replace' on a block. I need to keep the helper methods if I'm not careful.
        # The prompt asks to update init and getitem.
        # I should output the FULL class or be very specific.
        # I'll stick to replacing the logic parts.
        
        # 1. Geometric
        k = random.choice([0, 1, 2, 3]) 
        flip_axis = random.choice([None, 0, 1]) 
        
        colors = list(range(1, 10))
        random.shuffle(colors)
        color_map = {0: 0}
        for i, c in enumerate(colors):
            color_map[i+1] = c
            
        def transform(grid):
            g = np.array(grid)
            if k > 0:
                g = np.rot90(g, k)
            if flip_axis is not None:
                g = np.flip(g, axis=flip_axis)
            g_aug = np.zeros_like(g)
            for src, dst in color_map.items():
                g_aug[g == src] = dst
            return g_aug.tolist()

        new_task = {'train': [], 'test': []}
        for pair in task['train']:
            new_task['train'].append({
                'input': transform(pair['input']),
                'output': transform(pair['output'])
            })
        for pair in task['test']:
            new_task['test'].append({
                'input': transform(pair['input']),
                'output': transform(pair['output'])
            })
        return new_task

    def __getitem__(self, idx):
        task = self.tasks[idx]
        
        if self.mode == 'train' and self.augment:
            task = self.apply_augmentation(task)
            
        # Select query/target
        if self.mode == 'train':
            # Training Strategy: Leave-One-Out or similar
            train_pairs = task['train']
            all_pairs = train_pairs + task.get('test', []) # Use all available data
            
            if len(all_pairs) < 1: return torch.zeros(1)
            
            query_idx = random.randint(0, len(all_pairs)-1)
            query_pair = all_pairs[query_idx]
            
            # Support (rest)
            support_pairs = [p for i, p in enumerate(all_pairs) if i != query_idx]
            
        else: # mode == 'test'
            # STRICT Evaluation: Query is from 'test', Support is 'train'
            # We iterate over test pairs? 
            # For simplicity in DataLoader, we pick the FIRST test pair.
            # Ideally we'd return a list or have dataset expand tests.
            query_pair = task['test'][0] 
            support_pairs = task['train']
            
        # -- Prepare Tensors (Same as before) --
        
        # Limit support
        max_support = 3
        if len(support_pairs) > max_support:
            if self.mode == 'train':
                support_pairs = random.sample(support_pairs, max_support)
            else:
                support_pairs = support_pairs[:max_support] # Deterministic subset
            
        def to_tensor(g):
            p, m = self.pad_grid(g)
            return torch.from_numpy(p).long(), torch.from_numpy(m).float()
            
        support_grids = []
        support_masks = []
        
        for p in support_pairs:
            in_g, in_m = to_tensor(p['input'])
            out_g, out_m = to_tensor(p['output'])
            support_grids.append(torch.stack([in_g, out_g]))
            support_masks.append(torch.stack([in_m, out_m]))
            
        q_in_g, q_in_m = to_tensor(query_pair['input'])
        q_out_g, q_out_m = to_tensor(query_pair['output'])
        
        while len(support_grids) < max_support:
            z = torch.zeros((2, self.grid_size, self.grid_size), dtype=torch.long)
            zm = torch.zeros((2, self.grid_size, self.grid_size), dtype=torch.float)
            support_grids.append(z)
            support_masks.append(zm)
            
        support_grids = torch.stack(support_grids) 
        support_masks = torch.stack(support_masks)
        
        return {
            'support': support_grids,       
            'support_mask': support_masks,  
            'query_input': q_in_g,          
            'query_mask': q_in_m,           
            'target': q_out_g,              
            'target_mask': q_out_m          
        }

def get_loader(path, batch_size=16, mode='train', augment=True):
    shuffle = (mode == 'train')
    ds = ARCDataset(path, mode=mode, augment=augment)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=4, pin_memory=True)
