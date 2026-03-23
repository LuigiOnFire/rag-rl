import sys
import os

# Add project root to sys.path
sys.path.append(os.getcwd())

from src.oracle.calibrator import EnergyCalibrator

if __name__ == "__main__":
    calibrator = EnergyCalibrator()
    calibrator.run()
