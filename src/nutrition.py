"""Physical plausibility checks for recipe-grounded nutrition data.

This module validates externally grounded per-100g macros. It does not infer recipe
quantities or cooked yield from an image.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class NutritionValidation:
    valid: bool
    calculated_kcal: float
    relative_energy_error: float
    macro_sum_g: float
    errors: tuple[str, ...]


def validate_per_100g_nutrition(
    kcal: float,
    carbohydrate_g: float,
    protein_g: float,
    fat_g: float,
    energy_tolerance: float = 0.15,
) -> NutritionValidation:
    if not 0 <= energy_tolerance <= 1:
        raise ValueError("energy_tolerance must be between zero and one")
    values = [float(kcal), float(carbohydrate_g), float(protein_g), float(fat_g)]
    errors: list[str] = []
    if not all(math.isfinite(value) for value in values):
        errors.append("non_finite_value")
    if any(value < 0 for value in values if math.isfinite(value)):
        errors.append("negative_value")
    macro_sum = carbohydrate_g + protein_g + fat_g
    if macro_sum > 100.0 + 1e-6:
        errors.append("macros_exceed_100g")
    if kcal > 900.0 + 1e-6:
        errors.append("energy_exceeds_physical_bound")
    calculated = 4 * carbohydrate_g + 4 * protein_g + 9 * fat_g
    relative_error = abs(kcal - calculated) / max(kcal, calculated, 1.0)
    if math.isfinite(relative_error) and relative_error > energy_tolerance:
        errors.append("atwater_energy_mismatch")
    return NutritionValidation(
        valid=not errors,
        calculated_kcal=float(calculated),
        relative_energy_error=float(relative_error),
        macro_sum_g=float(macro_sum),
        errors=tuple(errors),
    )
