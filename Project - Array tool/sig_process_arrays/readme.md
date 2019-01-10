# Arrays

Plots the beampattern of 1D arrays (Signal processing)

![](https://www.mathworks.com/matlabcentral/mlc-downloads/downloads/submissions/55924/versions/5/previews/html/eigenbeam_02.png)

## Getting Started

pip install the module (see installing). 
Try the following in a python shell:

```
import sig_process_arrays as s

# Generates a default Uniform Linear Array object
a = s.ULA() 

# Calculates response to 77Ghz signal
a.plot_beampattern(77e9)
```

## Prerequisites
The code was built and tested on a machine with the following enviornment:

```
Windows 10
Python 3.6.6
Numpy 1.15.3
Matplotlib 3.0.0
```

### Installing

Run: ```pip install sig-process-arrays```

## Running the tests
Tests are same as the "getting started" part.
However, you can read the documentation of the functions
and define ULA's with different spacing and elements.

## Authors
**Sammy Kolpinizki**

## License
This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## TODO:
- Add descriptions about HPBW, ambiguities, etc.
- Improve plots.
- Implement the UCA (Uniform Circular Array)
- Implement a UPA. 

## Last words
The main purpose of this package is an exercise for
uploading packages to PyPi. (Udacity data science course).