"""Luce Spark: calibrated hot/cold residency for sparsely-activated models.

Tooling (this package) runs outside the daemon: calibrate expert placement from
real traffic, validate, and (research) train a pre-gate predictor. The engine
that does the hot/cold offload + bounded expert cache lives in ../../server/.
"""
__version__ = "0.1.0"
