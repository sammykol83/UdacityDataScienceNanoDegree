import numpy as np

class Array:

    def __init__(self, num_of_elements = 8, spacing_m = 1e-3,
                 az_res_deg = 0.5, az_span = np.array([-50, 50])):
        """ General 1D array of EQUALLY spaced elements (either UCA/ULA)

        Attributes:
        	d (float) the spacing (in meters) between the elements
        	n (int) number of elements in array
            az_res_deg (float) - Azimuth resolution[deg] we want to build this array with.
            az_span (1x2 numpy vector) - Depicts start/end angles for observation
            c (int) - Speed of light (m/sec)
        """
        self.d = spacing_m
        self.n = num_of_elements
        self.az_res_deg = az_res_deg
        self.az_span = az_span
        self.c = 299792458

    def __repr__(self):
        """Function to output the characteristics of the array
                Args:
                Returns:
                    string: characteristics of the array
                """
        return "Array of {} elements with element spacing of {}[m]".\
            format(self.n, self.d)

    def plot_beampattern(self, fc_hz):
        """Function to plot the beampattern of the array at angle 0.
                Args:
                    fc_hz( float ) - Carrier frequency of waveform arriving to the array.
                Returns:
        """

    def build_bf_matrix(self, fc_hz, lambda_m):
        """Function holder for implemented classes"""
        pass








