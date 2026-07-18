import { describe, expect, it } from 'vitest';
import {
  FUELS,
  arrheniusRate,
  boundStoichiometricReaction,
  energyLimitedEvaporationMass,
  exponentialDissipation,
  heatOfCombustionJPerMol,
  radiativeTemperatureExchange,
  radiativeCooling,
  sliceGaussianWeight,
} from './physics';

describe('paper equations', () => {
  it('computes the Fire-X empirical methane heat of combustion', () => {
    expect(heatOfCombustionJPerMol(1, 4, 0)).toBeCloseTo(-834_000, -2);
  });

  it('keeps Arrhenius rates finite and temperature-sensitive', () => {
    const cold = arrheniusRate(300, 0.2, 0.8, FUELS.methane);
    const hot = arrheniusRate(1400, 0.2, 0.8, FUELS.methane);
    expect(Number.isFinite(cold)).toBe(true);
    expect(hot).toBeGreaterThan(cold);
  });

  it('applies Horvath cooling and exponential dissipation monotonically', () => {
    expect(radiativeCooling(1200, 1 / 60, 3000, 1700)).toBeLessThan(1200);
    expect(exponentialDissipation(1, 0.25, 1 / 60)).toBeLessThan(1);
  });

  it('uses broad, symmetric cross-slice particle support', () => {
    expect(sliceGaussianWeight(0, 0.1)).toBe(1);
    expect(sliceGaussianWeight(0.2, 0.1)).toBeCloseTo(sliceGaussianWeight(-0.2, 0.1));
    expect(sliceGaussianWeight(0.4, 0.1)).toBeCloseTo(Math.exp(-1));
  });

  it('bounds methane combustion by its four-to-one oxygen mass requirement', () => {
    const extent = boundStoichiometricReaction(1, 2, 1, 4);
    expect(extent.fuelConsumed).toBe(0.5);
    expect(extent.oxygenConsumed).toBe(2);
  });

  it('never evaporates more liquid than is present', () => {
    const mass = 0.2;
    const liquidTemperature = 800;
    const ambientTemperature = 300;
    const boilingTemperature = 373;
    const specificHeat = 4184;
    const latentHeat = 2.26e6;
    const evaporated = energyLimitedEvaporationMass(
      mass,
      liquidTemperature,
      ambientTemperature,
      boilingTemperature,
      specificHeat,
      latentHeat,
    );
    expect(evaporated).toBeGreaterThan(0);
    expect(evaporated).toBeLessThanOrEqual(mass);
    const availableEnergy = mass * specificHeat * (liquidTemperature - ambientTemperature);
    const usedEnergy =
      evaporated * (specificHeat * (boilingTemperature - ambientTemperature) + latentHeat);
    expect(usedEnergy).toBeLessThanOrEqual(availableEnergy + 1e-8);
  });

  it('cools hot gas toward ambient with the Fire-X T⁴ radiation term', () => {
    const hot = radiativeTemperatureExchange(1400, 300, 1, 1005, 1 / 120);
    const ambient = radiativeTemperatureExchange(300, 300, 1, 1005, 1 / 120);
    expect(hot).toBeLessThan(1400);
    expect(ambient).toBeCloseTo(300);
  });
});
