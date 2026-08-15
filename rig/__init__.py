"""Development rigs for the validity layer. Not part of the shipped package.

``pyproject.toml`` packages ``gates*`` only, so nothing here is installed with
the library. These are the harnesses that let Gate 1 be exercised end to end —
engineer turn, verdict, feedback report, rewrite — without spending an LLM call
or a 45-minute training run.
"""
