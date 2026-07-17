# Meridian Freight Tariff — Internal Reference (fictional)

Grounded source material for the challenger. Invented, so no solver can recall the answer:
every question written from this has to be worked out from the rules below.

## Zones

Every consignment has exactly one destination zone, and the base charge is that zone's rate
times the consignment weight. There is no origin zone and no per-leg pricing.

| Zone | Name      | Base rate | Transit days |
|------|-----------|-----------|--------------|
| A    | Coastwise | 12 credits/kg | 2 |
| B    | Midland   | 19 credits/kg | 4 |
| C    | Highland  | 27 credits/kg | 7 |
| D    | Farside   | 41 credits/kg | 11 |

## Surcharges

- **Fragile handling**: +15% of the base charge, Zones A and B only. In Zones C and D fragile
  cargo is refused outright.
- **Cold chain**: flat 340 credits per consignment, plus 6 credits/kg.
- **Oversize**: any single item over 80 kg adds 22% to the base charge.
- **Rush**: halves the transit days (round up) and adds 90% to the *total* charge, including
  every other surcharge. Not available in Zone D.

## Discounts

- Consignments over 500 kg take 8% off the base charge, applied before any surcharge.
- The Anchor account takes 5% off the base charge. If the 500 kg discount also applies, Anchor
  is taken off the already-discounted charge; if it does not, Anchor comes off the base charge
  directly.
- Discounts never apply to the cold-chain flat fee.

## Rules

1. Surcharges are computed on the base charge *after* discounts, unless stated otherwise.
2. Rush is always applied last.
3. A consignment refused under the fragile rule cannot be re-routed through another zone.
4. Transit days are counted from the day *after* pickup.
5. The Anchor discount requires the consignment to exceed 200 kg; below that it is void.
