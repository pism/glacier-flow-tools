all: dem_smoothing.pyx setup.py
	python3 setup.py build_ext --inplace

cythonize: dem_smoothing.pyx
	cython3 -a $^
