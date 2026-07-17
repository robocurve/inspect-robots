"""In-process execution boundary for model-generated Python code.

Model output executes with the evaluator's process privileges. This module is
an integration surface, not a security sandbox; untrusted models require an
external container or equivalent isolation.
"""
