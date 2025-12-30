# Hybrid ReDSAR: Recursive Dual-Stream Active Reasoning
## Architecture Overview for ARC-AGI

**Hybrid ReDSAR** is a novel neural architecture designed to solve the **AbstraXion Reasoning Corpus (ARC-AGI)**. It combines the strengths of Convolutional Neural Networks (CNNs) for local spatial physics with Transformers for global abstract reasoning, unified in a recurrent "active thinking" loop.

### 1. Core Philosophy: System 1 & System 2
The model is built on the hypothesis that ARC tasks require two distinct types of processing:
*   **System 1 (Local Stream)**: Handles pixel-level physics, cellular automata rules, and spatial adjacency. Implemented via **CNNs**.
*   **System 2 (Global Stream)**: Handles abstract logic, counting, symmetry, and high-level rules. Implemented via **Transformers** (Attention).

These two streams run in parallel and communicate (fuse) at every reasoning step.

---

### 2. The HybridTRM Architecture

#### A. Dual-Stream Blocks
The recursive unit of the model is the `DualStreamBlock`.
1.  **Local Stream (CNN)**:
    *   Input: `(B, C, H, W)` grid state.
    *   Operation: ResNet-style Convolution -> InstanceNorm -> GELU.
    *   Role: Propagates local updates (moving pixels, filling regions).
2.  **Global Stream (Transformer)**:
    *   Input: `(B, N, D)` global latent tokens (Abstract Memory).
    *   Operation: Self-Attention -> MLP.
    *   Role: Maintains the "program" or "rule" of the task.
3.  **Fusion (Communication)**:
    *   **Local->Global**: The grid feature map is flattened and mean-pooled, then injected into the Global Stream via a linear projection.
    *   **Global->Local**: The global tokens are averaged and broadcast back onto the grid, adding "top-down" influence to pixel changes.

#### B. Context Priming (In-Context Learning)
Unlike standard models that simply concatenate demos, Hybrid ReDSAR explicitly **studies** the support examples first.
*   **`encode_support(demos)`**:
    *   The model runs on each Support Pair `(Input, Output)`.
    *   It evolves a `z_global` state representing the "rule" of that specific example.
    *   These states are aggregated (pooled) to form a `z_global_init`.
*   **Inference**:
    *   When solving the **Query**, the model starts with this `z_global_init` instead of random noise.
    *   This effectively "loads the program" learned from the demos into the working memory.

#### C. Recurrent Active Reasoning
The model does not output a solution in one pass. It iterates for `k` steps (default 4-12) using the *same* weights (Recurrent Neural Network approach).
*   **Step `t`**: The model produces updates to both the grid `z_local` and the memory `z_global`.
*   **Deep Supervision**: We calculate the loss at **every single step**, forcing the model to solve the problem as quickly as possible.
*   **Critic Head**: A designated head estimates the "correctness" of the current state, potentially allowing for early-exit inference or search strategies (e.g., "Think longer if unsure").

---

### 3. Training Strategy

*   **Objective**: minimizes CrossEntropy Loss (Reconstruction) + BCE Loss (Critic Score).
*   **Augmentations**: Critical for generalization.
    *   Random Rotations (0, 90, 180, 270).
    *   Random Flips (Horizontal, Vertical).
    *   Color Permutations (Swapping color identities).
*   **Optimization**:
    *   **Gradient Clipping**: Essential for deep recurrent models to prevent NaN.
    *   **Scaled Initialization**: Fusion layers initialized near zero to stabilize early training.

---

### 4. Technical Specs
*   **Framework**: PyTorch
*   **Dimensions**: Default `dim=256`, `layers=4` (spatial depth).
*   **Recurrence**: Default `steps=4` (temporal depth).
*   **Hardware**: Optimized for Single GPU (RTX 3090/4090).

### 5. Running
*   **Train**: `run_train.bat`
*   **Benchmark**: `inference.bat`

**H100**
python train.py --epochs 2000 --layers 12 --dim 512 --batch_size 8 --lr 1e-4 --data_path ARC-AGI/data/training
