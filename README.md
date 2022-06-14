# Autoencoder-based Attribute Noise Handling Method for Medical Data

This repository contains the source code necessary to reproduce the experiments from the paper ''Autoencoder-based Attribute Noise Handling Method for Medical Data''.

- The *data* folder contains 3 real-world incomplete medical datasets we used for our experiments.
- The *models* folder contains our autoencoder-based architectures and implementations of various models used for our experiments.
- The various notebooks contains all our experiments.
- The *run\_all.sh* script can be used to execute all notebooks at once in background.
- The *correction.py* file contains our method's implementation that can be used by calling the 'run' functions as showed in the notebooks.

## Dependencies

- Python 3+
- PyTorch
- numpy
- TensorFlow
- geomloss
- sklearn
- pandas
- matplotlib
- missingpy


## License

MIT License

Autoencoder-based Attribute Noise Handling Method for Medical Data

Copyright (c) 2022 Thomas RANVIER

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
