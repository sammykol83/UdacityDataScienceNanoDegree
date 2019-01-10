from Arrays.Array import Array
import numpy as np
import matplotlib.pyplot as plt


class ULA(Array):
    """Uniform Linear Array"""

    def __init__(self, num_of_elements=8, spacing_m=2e-3,
                 az_res_deg=0.5, az_span=np.array([-50, 50])):

        elements = np.arange(0, num_of_elements * spacing_m, spacing_m)
        self.elements = np.reshape(elements, (elements.shape[0], 1))
        Array.__init__(self, num_of_elements, spacing_m, az_res_deg, az_span)

    def calc_waveform_response(self, fc_hz, target_az_deg):
        """Returns complex array response for waveform of arbitrary angle of arrival
                    Args:
                        fc_hz( float ) - a vector of angles to calculate the beamforming upon.
                        target_az_deg (float) - The waveform's AOA. (Angle Of Arrival)
                    Returns:
                        1xP complex matrix of the array response, where we calculated it over 'P' angles"""
        # Init
        target_az_rad = target_az_deg * np.pi / 180
        lambda_m = self.c / fc_hz

        # Calculate the BF matrix for current frequency
        bf_mat = self.build_bf_matrix(fc_hz, lambda_m)

        # Generate the steering vector
        a = np.exp(1j * 2 * np.pi * self.elements / lambda_m * np.sin(target_az_rad))
        a = np.reshape(a, (a.shape[0], 1))

        # Beamforming
        bf_mat = np.conj(np.transpose(bf_mat))
        res = np.dot(bf_mat, a)

        return res

    def build_bf_matrix(self, fc_hz, lambda_m):
        """Function that builds the beamforming matrix of the array
            Args:
                fc_hz( float ) - a vector of angles to calculate the beamforming upon.
                lambda_m (float) - wavelength in meters
            Returns:
                a numpy matrix of N x P, where 'N' is the number of elements and 'P'
                the number of angles"""
        sin_theta = np.sin(np.pi/180 * np.arange(self.az_span[0], self.az_span[1], self.az_res_deg))
        sin_theta = np.reshape(sin_theta, (1, sin_theta.shape[0]))
        bf_mat = np.exp(1j * 2 * np.pi * self.elements / lambda_m * sin_theta)
        return bf_mat

    def plot_beampattern(self, fc_hz):
        """Plots the magnitude of the array response for waveform arriving at boresight
            Args:
                fc_hz( float ) - a vector of angles to calculate the beamforming upon.
            Returns:
        """
        res = self.calc_waveform_response(fc_hz, 0)
        res_db_normalized = 10*np.log10(np.abs(res/np.max(np.abs(res)))**2)
        az_vec = np.arange(self.az_span[0], self.az_span[1], self.az_res_deg)
        plt.plot(az_vec, res_db_normalized)
        plt.grid()
        plt.xlabel('Azimuth[$\circ$]')
        plt.ylabel('dB')
        plt.show()