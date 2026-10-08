## Current real-network placement · 5 October 2026

Reference **CURRENT-HYDRAULIC-DOMAIN-2026-10-05**. This section replaces the historical gauge-event overlay as the evidence for the actual 16 study networks. It uses 5,315 physical base links from completed production inputs: final slopes, BASED-anchored alpha=.65 depths, n=.035, and recorded raw/used celerities. The subsequent historical sections retain their dated settings.

Water velocity is either steady width-allocated Q/(W h), or raw Manning celerity times 3/5. **Capped routing celerity is not substituted for water velocity.** Peak-Q coordinates keep reference depth fixed; these are sensitivity scenarios, not routed internal peaks. Very high allocated-flow Froude values indicate reference-state inconsistency, not simulated high-speed flow.

The wave-regime coordinates depend on velocity, depth, slope and a perturbation period, as in [Getirana & Paiva (2013), equations 3–4](https://gmao.gsfc.nasa.gov/media/gmaoftp/sarith/ROUTING_MODEL/docs/amazon_wrcr20212.pdf). A faster routing result does not itself determine this placement. The green region remains the repository's provisional screen, not the complete literature boundary or proof of RAPID validity.

**Duration uncertainty is material.** We show first forcing 1% departure-to-first-peak (a broad event-window proxy) and the stored local-minimum-to-peak (a potentially very short shoulder proxy). Neither is adopted as the true perturbation period. Broad departures can span multiple rises, especially example 48. These alternatives do not change the accepted parent–child response metrics. Synthetic points use their prescribed 12 h rise; this differs slightly from a 1% departure for a triangular rise.

The broad-window figure below gives real baseflow passing-length fractions of **16.3% allocated / 17.6% Manning**, versus approximately **0.05% / 0.04%** with the stored local rise. Those are pooled physical-length fractions, not percentages of valid rivers. They demonstrate sensitivity, not a basis for choosing whichever duration passes more links. No matched old/new hydraulic reconstruction was made, so this does not establish improvement over the old parameterization.

**Interpretation:** updated synthetic and real coordinates overlap partially, while real networks occupy a much wider range. The previous gauge-screening cohort is not a matched historical baseline. Current results do not yet support “the faster model is now in the correct hydraulic domain.” Resolve a defensible event-period proxy and the Q/depth/Manning consistency diagnostics before making that claim. Keep agreement with observed hydrographs as a separate validation question.

Reproduce with `RAPID/investigations/current_hydraulic_domain.py`. Source identities and formulas are in [the audit provenance](../../RAPID/validation/current_hydraulic_domain_20261005/provenance.json).

No routing was rerun. Assertions check graph connectivity, conservation of steady width shares, finite diagnostics and raw-celerity parity with the saved production depth/slope at n=.035.
