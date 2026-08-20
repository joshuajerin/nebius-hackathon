"""Test collection boundaries for preserved historical prototypes.

These three files target an abandoned root-level, one-workcell API (including
``REGISTRY``/``VISION_CELL`` and 150/40 mm tag contracts).  They are retained
unchanged per the no-deletion rule.  The maintained direct-USB prototype is
covered by ``test_direct_usb_*``; the approved two-workbench implementation is
covered by ``test_approved_*`` and the navigation/docking suites.
"""

collect_ignore = [
    "test_contracts.py",
    "test_diagnosis.py",
    "test_mission.py",
    "test_target_runtime.py",
]
