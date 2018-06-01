# inline_holo
Python code for inline holography using through-focus image series.

The purpose of this repository is to develop code that adapts to the needs of focal series theoretical and experimental analysis, in optical and electron microscopy, taking advantage of specialized Python libraries such as hyperspy and pycuda.

Requirements
------------
It needs a working hyperspy environment. That includes other packages such as numpy, scipy, matplotlib, etc. These should also be included in that environment.

CUDA support is optional; it needs pycuda and skcuda modules to be installed and working. Activating the optional CUDA support in some modules can improve significantly the running time.

Usage
-----
```python
from inline_holo import *
```
Also, the provided jupyter notebook `Simulated_dataset.ipynb` can be used to learn how to use the code in a simulated dataset example.
