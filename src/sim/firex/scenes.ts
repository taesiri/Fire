export interface FireXScenePreset {
  readonly id: FireXSceneId;
  readonly label: string;
  readonly values: Readonly<Record<string, number | boolean>>;
  readonly expected: string;
}

export const FIREX_FIELD_VIEWS = Object.freeze([
  Object.freeze({ id: 'beauty', value: 0, label: 'Beauty', help: 'Composited flame, products, steam and droplets. Use a diagnostic field to verify the underlying state.' }),
  Object.freeze({ id: 'temperature', value: 1, label: 'Temperature', help: 'Thermal field. The hot core should rise with buoyancy and contract during effective base suppression.' }),
  Object.freeze({ id: 'reaction', value: 2, label: 'Reaction', help: 'Local combustion activity. It should remain near mixed fuel and oxygen, then weaken when water reaches the source.' }),
  Object.freeze({ id: 'fuel-oxygen', value: 3, label: 'Fuel / O₂', help: 'Reactants: fuel and oxygen should meet at the reaction zone and deplete in the expected stoichiometric direction.' }),
  Object.freeze({ id: 'products', value: 4, label: 'CO₂ / products', help: 'Combustion products and residual history should trail the active reaction region rather than lead it.' }),
  Object.freeze({ id: 'vapor-soot', value: 5, label: 'Vapor / soot', help: 'Steam and incomplete-combustion residue. Base spray should increase vapor while a rich burn increases residue.' }),
  Object.freeze({ id: 'liquid-velocity', value: 6, label: 'Liquid / velocity', help: 'Deposited liquid and gas motion. This view confirms that the nozzle field intersects the intended target height.' }),
  Object.freeze({ id: 'divergence', value: 7, label: 'Divergence', help: 'Projection diagnostic. A mostly dark field after projection indicates a better incompressibility solve.' }),
] as const);

export type FireXFieldViewId = (typeof FIREX_FIELD_VIEWS)[number]['id'];

export const FIREX_SCENE_IDS = [
  'methane',
  'large-fire',
  'inferno',
  'rich',
  'top-jet',
  'base-spray',
] as const;

export type FireXSceneId = (typeof FIREX_SCENE_IDS)[number];

export const FIREX_SCENE_PRESETS: readonly FireXScenePreset[] = Object.freeze([
  Object.freeze({
    id: 'methane',
    label: 'Methane',
    values: Object.freeze({ burnerSize: 1, burnerDepth: 1, sourceThickness: 1, firePower: 1, sourceLift: 1, fuelRate: 0.48, oxygenRate: 0.9, heatEfficiency: 0.78, vorticity: 0.9, buoyancyScale: 1, flamePersistence: 1, airEntrainment: 0, waterFlow: 0, sprayAngle: 10, aimHeight: 0.42, domainWidth: 1, domainHeight: 1, domainDepth: 1, viewZoom: 1, exposure: 1, frontDefinition: 1.15, nozzleType: 0, waterEnabled: false }),
    expected: 'A clean, vertically stable flame with a compact reaction zone and no liquid deposition.',
  }),
  Object.freeze({
    id: 'large-fire',
    label: 'Large fire',
    values: Object.freeze({ burnerSize: 2.25, burnerDepth: 2, sourceThickness: 1.25, firePower: 2.1, sourceLift: 1.8, fuelRate: 0.42, oxygenRate: 1, heatEfficiency: 0.92, vorticity: 1.4, buoyancyScale: 1.4, flamePersistence: 1.45, airEntrainment: 1.6, waterFlow: 0, sprayAngle: 10, aimHeight: 0.42, domainWidth: 1, domainHeight: 1, domainDepth: 1, viewZoom: 1.55, exposure: 0.85, frontDefinition: 1.45, nozzleType: 0, waterEnabled: false }),
    expected: 'A broad elliptical burner, stronger lift and tighter framing should produce a genuinely large flame.',
  }),
  Object.freeze({
    id: 'inferno',
    label: 'Inferno',
    values: Object.freeze({ burnerSize: 3.1, burnerDepth: 2.7, sourceThickness: 1.35, firePower: 2.5, sourceLift: 1.6, fuelRate: 0.3, oxygenRate: 1, heatEfficiency: 0.86, vorticity: 3.6, buoyancyScale: 2.15, flamePersistence: 1.45, airEntrainment: 1.65, waterFlow: 0, sprayAngle: 12, aimHeight: 0.42, domainWidth: 1.15, domainHeight: 1.35, domainDepth: 1.15, viewZoom: 2.1, exposure: 1.25, frontDefinition: 1.65, nozzleType: 0, waterEnabled: false }),
    expected: 'A near-domain-wide burner should form a large oxygen-fed, turbulent multi-tongue flame without striking the top boundary.',
  }),
  Object.freeze({
    id: 'rich',
    label: 'Rich',
    values: Object.freeze({ burnerSize: 1, burnerDepth: 1, sourceThickness: 1, firePower: 1, sourceLift: 1, fuelRate: 0.92, oxygenRate: 0.42, heatEfficiency: 0.62, vorticity: 1.6, buoyancyScale: 1, flamePersistence: 1, airEntrainment: 0, waterFlow: 0, sprayAngle: 12, aimHeight: 0.42, domainWidth: 1, domainHeight: 1, domainDepth: 1, viewZoom: 1, exposure: 1, frontDefinition: 1.35, nozzleType: 0, waterEnabled: false }),
    expected: 'Oxygen limitation should reduce clean reaction and leave more product or soot history above the source.',
  }),
  Object.freeze({
    id: 'top-jet',
    label: 'Top jet',
    values: Object.freeze({ burnerSize: 1, burnerDepth: 1, sourceThickness: 1, firePower: 1, sourceLift: 1, fuelRate: 0.65, oxygenRate: 0.82, heatEfficiency: 0.75, vorticity: 1.1, buoyancyScale: 1, flamePersistence: 1, airEntrainment: 0, waterFlow: 0.7, sprayAngle: 6, aimHeight: 0.64, domainWidth: 1, domainHeight: 1, domainDepth: 1, viewZoom: 1, exposure: 1, frontDefinition: 1.25, nozzleType: 0, waterEnabled: true }),
    expected: 'A narrow upper-plume jet should cool locally, but the reaction at the fuel source should largely persist.',
  }),
  Object.freeze({
    id: 'base-spray',
    label: 'Base spray',
    values: Object.freeze({ burnerSize: 1, burnerDepth: 1, sourceThickness: 1, firePower: 1, sourceLift: 1, fuelRate: 0.65, oxygenRate: 0.82, heatEfficiency: 0.75, vorticity: 1.2, buoyancyScale: 1, flamePersistence: 1, airEntrainment: 0, waterFlow: 0.74, sprayAngle: 30, aimHeight: 0.14, domainWidth: 1, domainHeight: 1, domainDepth: 1, viewZoom: 1, exposure: 1, frontDefinition: 1.25, nozzleType: 1, waterEnabled: true }),
    expected: 'A broad spray aimed at the base should contract the hot reaction field and create a rising vapor region.',
  }),
]);

export function getFireXScene(id: string): FireXScenePreset {
  const scene = FIREX_SCENE_PRESETS.find((candidate) => candidate.id === id);
  if (!scene) throw new RangeError(`Unknown Fire-X scene: ${id}`);
  return scene;
}

export function getFireXFieldView(id: string): (typeof FIREX_FIELD_VIEWS)[number] {
  const view = FIREX_FIELD_VIEWS.find((candidate) => candidate.id === id);
  if (!view) throw new RangeError(`Unknown Fire-X field view: ${id}`);
  return view;
}
