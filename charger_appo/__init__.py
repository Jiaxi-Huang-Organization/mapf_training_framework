"""
Charger APPO Module

Charger policy for MAPF with battery-aware navigation.

Key features:
- Battery-aware target switching (goal vs charger)
- Follower encoder weight loading and freezing
- Scalar feature processing (xy, target_xy, battery, charge_xy)
"""

__version__ = '1.0.0'
