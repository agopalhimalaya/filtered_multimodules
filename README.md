# Filtered Bordered Floer Computations for Torus-Boundary Manifolds

Python code for filtered bordered Floer computations for manifolds with torus boundary, with coefficients over $\mathbb F_2$ and formal variables $W_1,W_2,W_3,\ldots$. The multimodules themselves are the usual unfiltered hat-version bordered Floer multimodules, while the tensor-product routines are written to be compatible with the filtration on the torus-boundary invariants.

The codes are written over Jonathan Hanselman's `graph_manifolds_HFhat.py` and modified so that tensor products with the usual multimodules are compatible with the filtration. The code was also written with ChatGPT through iterative training and prompting on the codebase to make it more readable and help verify the implementations.

## Current scope

At present, the filtration is implemented only for bordered Floer invariants of manifolds with torus boundary, using the formal variables $W_1,W_2,\ldots$. The multimodules used in the tensor products are the usual unfiltered hat-version bordered Floer multimodules; the tensor-product code is modified to preserve and propagate the filtration carried by the torus-boundary invariant.

## Filtered torus-boundary invariants

For a manifold with torus boundary, we use a bordered Heegaard diagram with a basepoint $p$ on the boundary together with interior basepoints such as $w$ and $z$. The interior basepoints give formal variables and hence filtrations.

Our notation is:

- an **underlined** basepoint means that the corresponding formal variable is set equal to $0$;
- a **non-underlined** basepoint means that the object is filtered by the corresponding formal variable.

We use five related hat-version objects:

1. $g\widehat{CFD}(\underline{w},\underline{p})$, where the formal variables corresponding to $w$ and $p$ are set to $0$.

2. $f\widehat{CFD}(w,\underline{p})$, where the formal variable for $p$ is set to $0$, while the object is filtered by the formal variable corresponding to $w$.

3. $g\widehat{CFD}(\underline{w},\underline{z},\underline{p})$, where the formal variables corresponding to $w,z,p$ are all set to $0$.

4. The tube-cutting bimodule is $\widehat{CFDD}(\mathcal{Y}_{TC})$.

5. $f\widehat{CFD}(w,\underline{z},\underline{p})$, where the formal variables corresponding to $z$ and $p$ are set to $0$, while the object is filtered by the formal variable corresponding to $w$.

## Files

- `filtered_multimodules.py` — defines the `Multimodule` class and several standard bordered Floer modules.
- `tensormor.py` — computes box tensor products and filtered/unfiltered specializations.
- `display_latex.py` — displays type $D$, $DD$, $DDD$, etc. structures in LaTeX.

## Operation format

Operations are stored as

```python
[start, W_monomial, boundary_1, ..., boundary_n, end]
```

For example,

```python
["a", "", "1", "b"]
```

represents an operation with coefficient $\rho_1$, while

```python
["a", "W_1", "12", "b"]
```

represents an operation with coefficient $W_1\rho_{12}$.

The empty string in the $W$-coefficient position means coefficient $1$.

## Defining a multimodule

```python
from filtered_multimodules import Multimodule

M = Multimodule(
    "example",
    ["D"],
    ["a", "b"],
    {
        "a": [0],
        "b": [1],
    },
    [
        ["a", "W_1", "1", "b"],
    ],
)
```

The idempotent convention is

```text
0 = iota_0
1 = iota_1
```

## Box tensor product

Import

```python
from tensormor import tensor
```

and compute

```python
product = tensor(
    module1,
    boundary1,
    module2,
    boundary2,
    simplify=False,
)
```

The tensor product matches compatible generators and type $A$/type $D$ algebra data and multiplies the formal $W_i$-coefficients along matched paths.

By default, differential cancellation is not performed automatically.

## Differential cancellation

To cancel removable coefficient-$1$ differentials, use

```python
M.remove_differentials()
```

Only unit differentials are cancelled. A differential carrying a nontrivial $W_i$-coefficient is not treated as a cancellable unit differential.

Since cancellation modifies the module in place, use `deepcopy` if the unreduced module should also be kept:

```python
from copy import deepcopy

M_reduced = deepcopy(M)
M_reduced.remove_differentials()
```

## Associated graded and unfiltered versions

To set $W_1=W_2=\cdots=0$, use

```python
from tensormor import gradedW0

M_graded = gradedW0(M)
```

Operations containing a positive power of any $W_i$ disappear.

To obtain the unfiltered reduced version by setting $W_1=W_2=\cdots=1$, use

```python
from tensormor import unfiltered_reduced

M_unfiltered = unfiltered_reduced(M)
```

## Built-in modules

`filtered_multimodules.py` includes constructors such as

```python
fCFDsolidtorus_1()
fCFDsolidtorus_2()

CFAA_id()
CFDD_id()

tube_cutting_piece()

CFDA_twist1()
CFDA_twist1_inv()

CFDA_twist2()
CFDA_twist2_inv()

CFDDD()
CFDDD_mirror()
```

## LaTeX display

For modules whose remaining boundaries are all type $D$, use

```python
from display_latex import display_type_D_differential

display_type_D_differential(M)
```

This displays the generators, idempotents, and type $D$ differential in LaTeX form.

To return the LaTeX strings without displaying them,

```python
generators_latex, operations_latex = (
    display_type_D_differential(
        M,
        show=False,
    )
)
```
