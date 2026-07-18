export const AMBIENT_TEMPERATURE_K = 300;
export const UNIVERSAL_GAS_CONSTANT = 8.314;
export const STEFAN_BOLTZMANN = 5.67e-8;

export interface FuelChemistry {
  key: string;
  label: string;
  carbon: number;
  hydrogen: number;
  oxygen: number;
  molarMassKg: number;
  activationEnergyJ: number;
  preExponential: number;
  fuelExponent: number;
  oxygenExponent: number;
  stoichOxygenMassRatio: number;
}

export const FUELS: Record<string, FuelChemistry> = {
  methane: {
    key: 'methane',
    label: 'Methane',
    carbon: 1,
    hydrogen: 4,
    oxygen: 0,
    molarMassKg: 0.01604,
    activationEnergyJ: 1.25e5,
    preExponential: 8.3e5,
    fuelExponent: -0.3,
    oxygenExponent: 1.3,
    stoichOxygenMassRatio: 4.0,
  },
  propane: {
    key: 'propane',
    label: 'Propane',
    carbon: 3,
    hydrogen: 8,
    oxygen: 0,
    molarMassKg: 0.0441,
    activationEnergyJ: 1.25e5,
    preExponential: 8.6e11,
    fuelExponent: 0.1,
    oxygenExponent: 1.65,
    stoichOxygenMassRatio: 3.64,
  },
};

/** Fire-X Eq. 4, converted from kJ/mol to J/mol. */
export function heatOfCombustionJPerMol(
  carbon: number,
  hydrogen: number,
  oxygen: number,
): number {
  return -417_000 * (carbon + 0.25 * hydrogen - 0.5 * oxygen);
}

/** Fire-X Eq. 5 with guards suitable for finite-precision GPU evaluation. */
export function arrheniusRate(
  temperatureK: number,
  fuelConcentration: number,
  oxygenConcentration: number,
  chemistry: FuelChemistry,
): number {
  if (temperatureK <= 0 || fuelConcentration <= 0 || oxygenConcentration <= 0) return 0;
  const fuel = Math.max(fuelConcentration, 1e-8);
  const oxygen = Math.max(oxygenConcentration, 1e-8);
  return (
    chemistry.preExponential *
    Math.exp(-chemistry.activationEnergyJ / (UNIVERSAL_GAS_CONSTANT * temperatureK)) *
    Math.pow(fuel, chemistry.fuelExponent) *
    Math.pow(oxygen, chemistry.oxygenExponent)
  );
}

/** Horvath-Geiger Eq. 3. */
export function radiativeCooling(
  temperature: number,
  dt: number,
  coefficient: number,
  maximumTemperature: number,
): number {
  const ratio = Math.max(temperature, 0) / Math.max(maximumTemperature, 1e-6);
  return Math.max(0, temperature - dt * coefficient * ratio ** 4);
}

/** Horvath-Geiger Eqs. 4 and 5. */
export function exponentialDissipation(value: number, rate: number, dt: number): number {
  return value * Math.pow(Math.max(0, 1 - rate), dt);
}

/** Fire-X Eq. 2 radiative exchange term integrated for one explicit step. */
export function radiativeTemperatureExchange(
  temperatureK: number,
  ambientTemperatureK: number,
  emissivity: number,
  specificHeatJPerKgK: number,
  dt: number,
): number {
  const temperature = Math.max(temperatureK, 0);
  const ambient = Math.max(ambientTemperatureK, 0);
  const heatCapacity = Math.max(specificHeatJPerKgK, 1e-8);
  const derivative =
    (Math.max(emissivity, 0) * STEFAN_BOLTZMANN * (ambient ** 4 - temperature ** 4)) /
    heatCapacity;
  return Math.max(0, temperature + Math.max(dt, 0) * derivative);
}

/** Horvath-Geiger Eq. 2. */
export function sliceGaussianWeight(distance: number, sliceSpacing: number): number {
  const sigma = Math.max(4 * sliceSpacing, 1e-6);
  return Math.exp(-(distance * distance) / (sigma * sigma));
}

export interface ReactionExtent {
  fuelConsumed: number;
  oxygenConsumed: number;
}

/** Bounds a Fire-X one-step reaction so no reactant mass can become negative. */
export function boundStoichiometricReaction(
  fuelMass: number,
  oxygenMass: number,
  requestedFuelConsumption: number,
  oxygenToFuelMassRatio: number,
): ReactionExtent {
  const fuelAvailable = Math.max(fuelMass, 0);
  const oxygenAvailable = Math.max(oxygenMass, 0);
  const ratio = Math.max(oxygenToFuelMassRatio, 1e-8);
  const fuelConsumed = Math.min(
    Math.max(requestedFuelConsumption, 0),
    fuelAvailable,
    oxygenAvailable / ratio,
  );
  return { fuelConsumed, oxygenConsumed: fuelConsumed * ratio };
}

/** Fire-X Eqs. 26–28, expressed as an energy-limited liquid mass transfer. */
export function energyLimitedEvaporationMass(
  availableLiquidMass: number,
  liquidTemperatureK: number,
  ambientTemperatureK: number,
  boilingTemperatureK: number,
  specificHeatJPerKgK: number,
  latentHeatJPerKg: number,
): number {
  const sensibleEnergy =
    Math.max(liquidTemperatureK - ambientTemperatureK, 0) *
    Math.max(specificHeatJPerKgK, 0) *
    Math.max(availableLiquidMass, 0);
  const energyPerKg =
    Math.max(specificHeatJPerKgK, 0) *
      Math.max(boilingTemperatureK - ambientTemperatureK, 0) +
    Math.max(latentHeatJPerKg, 0);
  if (energyPerKg <= 0) return 0;
  return Math.min(Math.max(availableLiquidMass, 0), sensibleEnergy / energyPerKg);
}
