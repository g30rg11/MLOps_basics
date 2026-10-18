def amount_of_substance(mass: float, molar_mass: float) -> float:
    """Calculate amount in moles from mass in g and molar mass in g/mol."""
    if mass < 0:
        raise ValueError("Mass must be non-negative")
    if molar_mass <= 0:
        raise ValueError("Molar mass must be positive")

    return mass / molar_mass


def main() -> None:
    mass = 18.015
    molar_mass = 18.015
    amount = amount_of_substance(mass, molar_mass)

    print(f"Mass: {mass} g")
    print(f"Molar mass: {molar_mass} g/mol")
    print(f"Amount of substance: {amount:.3f} mol")


if __name__ == "__main__":
    main()
