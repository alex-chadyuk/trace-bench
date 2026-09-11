"""Bundled correlator: raw heterogeneous log feed -> parent-linked sequence views.

A clean-room implementation of the documented correlation model (identity keys
vs connector keys, one propagation round, attribution at the strongest reachable
level) extended with parent-link inference so the views carry `parent_pos`.
"""
