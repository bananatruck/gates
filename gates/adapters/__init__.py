"""Reference adapters wiring G.A.T.E.S. into specific host scaffolds.

Every assumption about a host system lives here. The gate core imports nothing
from this subpackage, and no adapter imports anything from a host scaffold —
each one is called *by* the scaffold, not the other way around. Porting to
AI-Scientist-v2 or another harness means adding one module alongside
``agentlab.py``.
"""
