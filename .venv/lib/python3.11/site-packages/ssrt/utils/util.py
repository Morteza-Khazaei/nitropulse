import numpy as np



def toLambda(f):
    """
    Converts frequency in GHz to wavelength in meters.
    
    Returns:
        float
            Wavelength in meters.
    """
    c0 = 299792458.  # speed of light [m/s]
    f_Hz = f * 1e9   # frequency in Hz
    return c0 / f_Hz


def toPower(sig_db):
    """
    Converts a value in dB to its linear power equivalent.
    
    Parameters:
        x : float
            Value in dB.
    
    Returns:
        float
            Linear power equivalent.
    """
    return 10 ** (sig_db / 10)


def toDB(sig_lin):
    """
    Converts a linear power value to dB.
    
    Parameters:
        x : float
            Linear power value.
    
    Returns:
        float
            Value in dB.
    """
    return 10 * np.log10(sig_lin)