# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Plug the router into the API server's auto-route middleware.

The middleware itself (``deltaserve.route_middleware``) is routing-agnostic and
also serves the routing-off case; this module is the routing-ON wiring: a
``RouterClient`` on the router service's socket as the route function.
"""

from deltaserve.route_middleware import (  # noqa: F401 — re-exported
    ROUTED_PATHS,
    AdapterRouteMiddleware,
    extract_text,
    install_middleware,
)


def install(app, cfg) -> None:
    """Called by the plugin, only with adapter routing on."""
    from deltaserve.adapter_routing.service import RouterClient, socket_path
    from deltaserve.adapter_routing.stages import index_dir

    client = RouterClient(socket_path(index_dir(cfg)),
                          timeout_s=cfg.route_timeout_ms / 1e3)
    install_middleware(app, client.route, cfg.route_model_name)
