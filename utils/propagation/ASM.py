import torch
import argparse
from odak.learn.wave import propagate_beam



def ASM(phase_profile: torch.Tensor, args: argparse.Namespace) -> torch.Tensor:
    wavelength = args.wavelength[0]
    result = propagate_beam(phase_profile, wavenumber(wavelength), args.propagation_distance, args.dx, wavelength, propagation_type='Angular Spectrum')
    return result

def generate_complex_field(amplitude: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
    field = amplitude*torch.cos(phase)+1j*amplitude*torch.sin(phase)
    return field

def wavenumber(wavelength: float) -> float:
    k = 2*torch.pi/wavelength
    return k