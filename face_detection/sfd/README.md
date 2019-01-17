Sourced from https://github.com/clcarwin/SFD_pytorch

# Note: to use cython without special C libraries, we can just call pyximport
# See https://cython.readthedocs.io/en/latest/src/tutorial/cython_tutorial.html


# S³FD: Single Shot Scale-invariant Face Detector
A PyTorch Implementation of Single Shot Scale-invariant Face Detector.

## Model
[s3fd_convert.7z](https://github.com/clcarwin/SFD_pytorch/releases/tag/v0.1)

## Test
```
python test.py --model data/s3fd_convert.pth --path data/test01.jpg
```

# References
[SFD](https://github.com/sfzhang15/SFD)
