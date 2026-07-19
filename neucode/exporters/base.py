"""
Abstract base class for NeuCoDe model exporters.
"""
from abc import ABC, abstractmethod
import math

class BaseExporter(ABC):
    """
    Abstract base class for fixed-point model exporters.

    Subclasses implement export() to serialise a trained model as a C header
    file using the shared Q-format quantisation helpers defined here.
    """
    @abstractmethod
    def export(self, model, filepath):
        """
        Export a trained model to the given file path.

        :param model: The trained model object to export.
        :param filepath: Destination path for the generated output file.
        """
        pass

    def _get_q_format(self, max_abs_value):
        """
        Calculate the Q-format parameters for fixed-point representation.

        :param max_abs_value: The maximum absolute value of the data.
        :returns: A tuple (m, n, scale) where m is the number of integer bits,
            n is the number of fractional bits, and scale is the quantisation
            scaling factor (2^n).
        """
        if max_abs_value == 0:
            return 0, 0, 0.0

        # Calculate Integer bits (m), this is derived from 
        # http://www.digitalsignallabs.com/downloads/fp.pdf
        # Section 4.4 - Signed Range
        # Range x of A(a,b) is: -2^(a) to 2^(a) - 2^(-b)
        # with a = m-1 (sign bit included), b = n: -2^(m-1) to 2^(m-1) - 2^(-n)
        # So we need m bits such that 2^(m-1) > max_abs_value
        # Ensure at least 1 integer bit (includes sign bit).
        # Without this clamp, values in (0, 0.5] yield m<=0 and can cause severe saturation.
        m = max(1, math.ceil(math.log2(max_abs_value)) + 1)

        # Calculate Fractional bits (n)
        n = self.total_bits - m

        scale = 2 ** n
        return m, n, scale