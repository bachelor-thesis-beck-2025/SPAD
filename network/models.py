# Temporary module to handle checkpoint compatibility
# Import TransferModel from the actual location
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from networks.xception_net import TransferModel

# Make it available as network.models.TransferModel for checkpoint loading
__all__ = ['TransferModel']
