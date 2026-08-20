# Required physical CAD

## Simulation-ready prototype

`so101_circular_service_effector.usda` is the current circular magnetic
service-puck asset. It replaces the visible SO-101 finger gripper in the
composed simulation and provides:

- a 110 mm diameter cylindrical magnet housing;
- a 104 mm diameter blue magnetic contact face;
- collision geometry owned by the SO-101 terminal rigid link;
- a named `ContactFrame` with its normal along local +X;
- headless-safe `UsdPreviewSurface` materials.

It is a dimensioned simulation prototype, not manufactured-tool CAD. Replace
it with measured geometry, mass, inertia, coil thermal limits, compliance, and
force-sensor data before physical deployment.

Do not replace these parts with visually similar Isaac props. Export the exact manufactured geometry as STEP and USD, with meters as the stage unit:

- Go2 top plate and SO-101 fixed mount, including fasteners;
- SICK InspectorP61x wrist bracket;
- spring-loaded or remote-center-compliance USB-C plug holder;
- replaceable USB-C tip, protective bezel and cable strain relief;
- workbench calibration plate carrying the 40 mm fine AprilTag and USB-C receptacle;
- controlled cable loop at the arm wrist.

Each USD needs collision geometry, mass, center of mass, inertia, material/friction metadata, named mount/tool frames, and a source drawing revision. The current procedural geometry is only a collision proxy and is intentionally reported as such by the scene scenario.
