# Factory SRE three-bundle imports

The Antioch catalog records are immutable JSON manifests. Inside a booted
Isaac Sim 6.0.1 process, use the Python registry to resolve their referenced
USDs and authored geometry.

```python
from factory_sre.sim.asset_bundles_v1 import BundleId, load_bundle

mobile = load_bundle(
    BundleId.MOBILE_MANIPULATOR,
    "/World/Robot",
)
workcell = load_bundle(
    BundleId.WORKCELL,
    "/World/Cell37",
)
warehouse = load_bundle(
    BundleId.WAREHOUSE,
    "/World/Warehouse",
)
```

Load the complete development scene with the same three loaders:

```python
from factory_sre.sim.asset_bundles_v1 import load_factory_sre_scene

reports = load_factory_sre_scene("/World/FactorySRE")
```

Each call requires a new, absolute, unused prim path. Stock Isaac assets are
referenced from the Isaac 6.0 asset root; SO-101 is loaded from
`so101_antioch@1.3.2`. The returned `BundleLoadReport` contains every created
prim path, resolved source, readiness verdict, and physical-calibration
warning.

The mobile-manipulator and workcell defaults are simulation prototypes and
return `deployment_ready=False`. Supply measured configuration values and set
their calibration flags only after the physical CAD and transforms have been
verified.
