import ULA

# Print the beampattern of the default 1x8 ULA at automotive radars frequencies
a = ULA()
a.plot_beampattern(77e9)
