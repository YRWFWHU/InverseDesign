import torch
import argparse
from utils.propagation.ASM import ASM
from utils.mask_utils import create_circle_mask


def phase_profile_mask_factory(phase_profile: torch.Tensor, phase_profile_mask_name: str) -> torch.Tensor:
    if phase_profile_mask_name == 'rect':
        return torch.ones_like(phase_profile)
    elif phase_profile_mask_name == 'circle':
        return create_circle_mask(phase_profile)
    else:
        raise ValueError(f"Invalid phase profile mask: {phase_profile_mask_name}")


def propagation_function_factory(propagation_function_name: str) -> callable:
    if propagation_function_name == 'default':
        return default_propagation
    else:
        raise ValueError(f"Invalid propagation function: {propagation_function_name}")


def default_propagation(phase_profile: torch.Tensor, args: argparse.Namespace) -> torch.Tensor:
    complex_field = ASM(phase_profile, args)
    intensity = torch.abs(complex_field)**2
    return intensity