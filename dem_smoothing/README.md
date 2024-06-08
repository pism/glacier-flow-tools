Python module that can be used to smooth ice surface elevation DEMs using Gaussian and triangular filters following

> F. S. McCormack, J. L. Roberts, L. M. Jong, D. A. Young, and L. H. Beem, “A note on digital elevation model smoothing and driving stresses,” Polar Research, vol. 38, no. 0, Mar. 2019, doi: 10.33265/polar.v38.3498.

Requires Cython and NumPy. Uses OpenMP to take advantage of all available CPU cores.

Run `python3 setup.py build_ext --inplace` to build and see documenting strings for more info.
