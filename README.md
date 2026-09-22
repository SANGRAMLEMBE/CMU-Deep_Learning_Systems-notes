# CMU Deep Learning Systems — Notes and Code

Working through **CMU 10-414/714, Deep Learning Systems: Algorithms and Implementation**
(Zico Kolter & Tianqi Chen), one notebook per lecture, alongside written notes.

Each notebook implements the lecture's concepts from scratch in NumPy and **checks the result**
rather than asserting it. No PyTorch, no TensorFlow — the point of the course is to build those,
not import them.

Sangram Lembe · [LinkedIn](https://www.linkedin.com/in/sangram-lembe-56262320a/) ·
[Portfolio](https://sangram-portfolio.onrender.com/) · sangramlembe@gmail.com

---

## Contents

| # | Notebook | What it builds and verifies |
|---|---|---|
| 1 | `Lecture_01_Why_Deep_Learning_Systems.ipynb` | The three pillars of a framework. Measures the loop-vs-matmul gap (**~46×**), then trains a classifier in six lines |
| 2 | `Lecture_02_Softmax_Regression.ipynb` | Softmax regression from scratch. Shows 0/1 loss is flat, that naive softmax returns `nan` where the stable form returns the right answer, and gradient-checks ∇θ = Xᵀ(Z − I_y) |
| 3 | `Lecture_03_Nonlinear_Hypothesis_Classes.ipynb` | Proves stacked linear layers collapse. Untrained random Fourier features take the rings from 70% error to 0.3%. Builds universal approximation explicitly from ReLUs |
| 4 | `Lecture_04_Backpropagation.ipynb` | Two-layer and L-layer backprop, both gradient-checked to ~1e-11. Measures the activation-memory cost |
| 5 | `Lecture_05_Automatic_Differentiation.ipynb` | All four ways to get a gradient — numerical, symbolic, forward mode, reverse mode — on one worked example, compared |
| 6 | `Lecture_06_Building_an_Autodiff_Framework.ipynb` | A miniature autodiff framework (~150 lines), then an MLP trained with **no hand-derived gradients** |

## Results worth pointing at

- **Lecture 1** — a Python loop and a matrix multiply compute identical numbers; the matmul is
  ~46× faster. Batch form is a performance decision, not notation.
- **Lecture 2** — 0/1 error is unchanged across four orders of magnitude of parameter
  perturbation. That is why it cannot be trained on.
- **Lecture 3** — a linear model gets ~70% error on concentric rings. A *fixed, untrained*
  random feature map drops it to 0.3%.
- **Lecture 4** — the L-layer backprop recursion agrees with finite differences to ~4e-11.
- **Lecture 5** — reverse mode recovers both partial derivatives in one backward pass; forward
  mode needs one pass per input.
- **Lecture 6** — the framework computes f''(2) = 12 for f(x) = x³, and trains an MLP to the
  same 2.2% test error as the hand-derived version in lecture 4.

## Dataset

The lectures use MNIST (28×28, so `n = 784`, 60,000 rows). These notebooks use scikit-learn's
bundled **digits** dataset — same task, 8×8 images, `n = 64`, ~1,800 rows — so they run anywhere
with no download.

The maths, shapes and gradients are identical. **Error rates here are not comparable** to the
"under 8% on MNIST" figure quoted in the lectures.

## Running them

```bash
pip install numpy scikit-learn matplotlib jupyter
jupyter notebook
```

Every notebook is self-contained and runs top to bottom in under a minute. All outputs are
committed, so they can be read on GitHub without running anything.

## A note on the checks

Kolter's own warning in lecture 2 is that the matrix-gradient derivation is a *shape argument*,
not a proof — it says which arrangement could work, not which one does. So every gradient in
these notebooks is verified against two-sided finite differences before it is used, and the test
shapes are deliberately unequal, because with `k == n` a wrong transpose still multiplies cleanly
and the bug hides.
