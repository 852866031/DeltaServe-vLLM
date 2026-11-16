# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Adapter routing (ARIADNE centroids): everything behind ``adapter_routing.enable_adapter_routing``.

Import boundary: nothing outside this package imports it at module level. The
plugin reaches it only through call sites guarded by
``vllm_config.adapter_routing_config.enable_adapter_routing``, so a server launched with
routing off never loads this package (``tests/test_routing_switch.py`` checks
that). The switch itself, ``AdapterRoutingConfig``, lives in ``deltaserve.config``.

The sentence encoder runs on plain ``transformers`` (already a vLLM dependency).
"""
