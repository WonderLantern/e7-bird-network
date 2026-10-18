Public numerical experiments
============================

The stripe and random inputs are procedural, not photographs. Their defined
retained-work and logical-head reference arrays are copied from the runtime
fixtures without including unrelated RAM or device-session buffers.

From a source checkout, run with --runtime .. while inside experiments/.
For the two extracted release archives, use --runtime ..\E7-Bird-Network.

  python -B generate_inputs.py --runtime .. --output ..\..\generated-inputs
  python -B verify_evidence.py --runtime .. --output ..\..\evidence-report.json
  python -B counterfactuals.py --runtime .. --case random --output ..\..\counterfactual-results

counterfactuals.py changes input division, product rounding/FTZ, guard bits,
carry arithmetic, bias order and reduction order. It defaults to the public
random fixture. Use --variants control for a quick control-only check.

historical-counterfactual-summary.json contains metrics from an earlier private
photographic input. Its image and all resulting tensors are deliberately absent.
Those historical numbers are not presented as results on the public random input.
No vendor documentation, SDK, firmware or device-session material is included.
