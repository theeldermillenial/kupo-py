"""FastAPI server for load balancing across multiple Kupo instances.

This module provides a proxy server that distributes requests across multiple
Kupo instances using round-robin load balancing, with optional DBSync fallback.
"""

import os
from contextlib import asynccontextmanager
from functools import partial
from typing import Union
from urllib.parse import urlparse
import asyncio
import logging

import aiohttp as aio
from aiohttp import ClientResponseError
from fastapi import FastAPI, Request, Response, HTTPException

from kupo.client import KupoClient, Order
from kupo.models import RoundRobin, KupoServer, DbsyncServer

# Optional DBSync dependency
try:
    import dbsync
    from dbsync.config import get_database_url
    from kupo.dbsync import get_matches as dbsync_get_matches

    DBSYNC_AVAILABLE = True
except ImportError:
    DBSYNC_AVAILABLE = False
    dbsync = None
    get_database_url = None
    dbsync_get_matches = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("kupo.server")
logger.setLevel(os.getenv("KUPO_PY_LOG_LEVEL", "INFO"))

# Global variables
check_task = None
kupo_servers: list[KupoServer] = []
dbysync_servers: list[DbsyncServer] = []
kupo_clients: list[KupoClient] = []

round_robin = RoundRobin(
    kupo_servers=kupo_servers,
    dbsync_servers=dbysync_servers,
)

ROLLBACK_SLOT = int(os.getenv("KUPO_PY_ROLLBACK_SLOT", "0"))
CHECK_INTERVAL = int(os.getenv("KUPO_PY_CHECK_INTERVAL", "1"))
PATTERN_SYNC_INTERVAL = int(os.getenv("KUPO_PY_PATTERN_SYNC_INTERVAL", "60"))


def verify_dbsync_connection():
    """Run a simple query to verify DBSync connection."""
    if not DBSYNC_AVAILABLE or dbsync is None:
        return

    with dbsync.session.get_session_context() as session:
        raw_conn = session.connection().connection
        cursor = raw_conn.cursor()
        try:
            cursor.execute("SELECT 1")
            cursor.fetchone()
        finally:
            cursor.close()


async def verify_servers():
    """Verify and populate servers at startup."""
    global round_robin  # Update global to refresh cycles

    # 1. Kupo Instances
    temp_kupo_servers = []
    temp_kupo_clients = []

    for key, value in os.environ.items():
        if key.startswith("KUPO_PY_URL"):
            try:
                # Ensure protocol exists for aiohttp compatibility
                if not value.startswith(("http://", "https://")):
                    value = f"http://{value}"

                url = urlparse(value)
                port = url.port or 1442
                hostname = url.hostname or "localhost"
                scheme = url.scheme or "http"

                client = KupoClient(base_url=f"{scheme}://{hostname}", port=port)

                # Test connection
                health_status = await client.health_async()

                server = KupoServer(url=value, health=health_status)
                temp_kupo_servers.append(server)
                temp_kupo_clients.append(client)
                logger.info("Added Kupo instance: %s", value)
            except Exception as e:
                logger.warning(
                    "Failed to connect to Kupo instance %s: %s. Skipping.",
                    value,
                    e,
                )

    # Update kupo_clients global in place
    kupo_clients[:] = temp_kupo_clients

    # 2. DBSync Instance
    temp_dbsync_servers = []
    if DBSYNC_AVAILABLE and get_database_url:
        try:
            # Get config
            dbsync_url = get_database_url()

            # Test connection
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, verify_dbsync_connection)

            server = DbsyncServer(url=dbsync_url)
            temp_dbsync_servers = [server]
            logger.info("Added DBSync instance from config.")

        except Exception as e:
            logger.warning(
                "Failed to connect to DBSync or load config: %s. "
                "Skipping DBSync fallback.",
                e,
            )
            # temp_dbsync_servers remains empty
    else:
        logger.info("DBSync library not installed. Skipping DBSync fallback.")

    # Re-create RoundRobin to ensure internal cycles are initialized
    round_robin = RoundRobin(
        kupo_servers=temp_kupo_servers,
        dbsync_servers=temp_dbsync_servers,
    )


async def check_health():
    """Check health of both Kupo instances using HTTP status codes."""
    logger.debug("Performing health check of Kupo instances")

    # Bulk check health of all clients
    try:
        health_checks = await asyncio.gather(
            *[client.health_async() for client in kupo_clients]
        )

        for s, health in zip(round_robin.kupo_servers, health_checks):
            logger.debug(
                "Health check result for %s: %s",
                s.url,
                health.status_code == 200 and not health.syncing,
            )
            s.health = health

        return

    except ClientResponseError as e:
        logger.error(
            "Error checking health of Kupo instances, "
            "checking individual instances: %s",
            e,
        )

    # If the bulk check fails, check individual instances
    for client, s in zip(kupo_clients, round_robin.kupo_servers):
        try:
            health = await client.health_async()
            logger.debug(
                "Health check result for %s: %s",
                s.url,
                health.status_code == 200 and not health.syncing,
            )
            s.health = health
        except ClientResponseError as e:
            logger.warning("Error checking %s instance: %s", s.url, e)
            s.health = None


async def sync_patterns():
    """Synchronize patterns between Kupo instances."""

    # Only sync if there is more than one instance and all are healthy
    if len(round_robin.kupo_servers) > 1:
        if None in [s.health for s in round_robin.kupo_servers]:
            return
        if any(s.health.syncing for s in round_robin.kupo_servers):
            return

    try:
        patterns = await asyncio.gather(
            *[client.get_patterns_async() for client in kupo_clients]
        )

        if len(patterns) < 2:
            return

        # Calculate union of all patterns
        instance_patterns = [set(p) for p in patterns]
        all_patterns = set().union(*instance_patterns)

        # Sync missing patterns
        for client, pattern in zip(kupo_clients, patterns):
            missing = all_patterns - pattern
            if missing:
                logger.info(
                    "Syncing %d missing patterns to %s",
                    len(missing),
                    client.base_url,
                )

                try:
                    await client.bulk_add_pattern_async(
                        list(missing), rollback_to=ROLLBACK_SLOT
                    )
                except ClientResponseError as e:
                    logger.error(
                        "Failed to sync patterns to %s: %s", client.base_url, e
                    )

    except ClientResponseError as e:
        logger.error("Error in pattern sync: %s", e)


async def check_periodically():
    """Periodically check health and sync patterns."""
    count = 0
    while True:
        await check_health()
        count += 1
        count %= PATTERN_SYNC_INTERVAL
        if count == 0:
            await sync_patterns()
        await asyncio.sleep(CHECK_INTERVAL)


@asynccontextmanager
async def lifespan(_fastapi_app: FastAPI):
    await verify_servers()

    global check_task
    check_task = asyncio.create_task(check_periodically())
    yield
    if check_task:
        check_task.cancel()
        try:
            await check_task
        except asyncio.CancelledError:
            pass


app = FastAPI(title="Kupo Router", lifespan=lifespan)


@app.get("/router/status")
async def router_status():
    """Get the status of the router and all Kupo instances."""
    return {
        "kupo_servers": [
            {
                "url": s.url,
                "health": s.health.model_dump() if s.health else None,
            }
            for s in round_robin.kupo_servers
        ],
        "dbsync_servers": [s.url for s in round_robin.dbsync_servers],
    }


async def handle_dbsync_matches(request: Request, path: str):
    """Handle match requests using DBSync as the backend."""
    if not DBSYNC_AVAILABLE or dbsync_get_matches is None:
        raise HTTPException(
            status_code=501,
            detail="DBSync support not installed",
            headers={"X-Kupo-Source": "dbsync"},
        )

    if path == "matches":
        pattern = "*"
    elif path.startswith("matches/"):
        pattern = path[len("matches/") :]
        if not pattern:
            pattern = "*"
    else:
        pattern = "*"

    qp = request.query_params

    try:
        order_str = qp.get("order", "most_recent_first")
        try:
            order = Order(order_str)
        except ValueError:
            order = Order.DESC

        loop = asyncio.get_running_loop()
        func = partial(
            dbsync_get_matches,
            pattern=pattern,
            spent="spent" in qp,
            unspent="unspent" in qp,
            resolve_hashes="resolve_hashes" in qp,
            order=order,
            created_after=qp.get("created_after"),
            spent_after=qp.get("spent_after"),
            created_before=qp.get("created_before"),
            spent_before=qp.get("spent_before"),
            policy_id=qp.get("policy_id"),
            asset_name=qp.get("asset_name"),
            transaction_id=qp.get("transaction_id"),
            output_index=(int(qp["output_index"]) if "output_index" in qp else None),
        )

        response = await loop.run_in_executor(None, func)

        return Response(
            content=response.model_dump_json(),
            media_type="application/json",
            headers={"X-Kupo-Source": "dbsync"},
        )
    except Exception as e:
        logger.error("DBSync execution error: %s", e)
        raise HTTPException(
            status_code=500, detail=str(e), headers={"X-Kupo-Source": "dbsync"}
        ) from e


@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def proxy_request(request: Request, path: str):
    """Proxy all requests to the active Kupo instance."""

    # Check for pattern updates
    if path == "patterns" and request.method in ["POST", "PUT"]:
        # Reject if only 1 active instance is available
        if len(round_robin.kupo_servers) > 1:
            healthy_count = sum(
                1
                for s in round_robin.kupo_servers
                if s.health is not None and not s.health.syncing
            )
            if healthy_count < 2:
                raise HTTPException(
                    status_code=503,
                    detail="Cannot update patterns: "
                    "Need at least 2 healthy Kupo instances.",
                )

    # Determine target backend
    force_backend = request.headers.get("X-Force-Backend")
    if force_backend:
        logger.debug("Force backend: %s", force_backend)
        force_backend = force_backend.lower()

    target_server: Union[KupoServer, DbsyncServer, None] = None

    if force_backend == "dbsync":
        if round_robin.dbsync_servers:
            target_server = round_robin.dbsync_servers[0]
    elif force_backend == "kupo":
        target_server = round_robin.next()
        if isinstance(target_server, DbsyncServer):
            target_server = None
    else:
        target_server = round_robin.next()

    if target_server is None:
        raise HTTPException(status_code=503, detail="No available server")

    # Handle DBSync
    if isinstance(target_server, DbsyncServer):
        logger.debug("Routing request to DBSync: %s", path)
        if path.startswith("matches"):
            return await handle_dbsync_matches(request, path)
        else:
            raise HTTPException(
                status_code=501,
                detail="DBSync fallback only supports matches",
                headers={"X-Kupo-Source": "dbsync"},
            )

    # Handle Kupo
    try:
        logger.debug("Routing request to Kupo (%s): %s", target_server.url, path)
        method = request.method
        headers = dict(request.headers)
        body = await request.body()
        headers.pop("host", None)
        headers.pop("content-length", None)

        async with aio.ClientSession() as session:
            async with session.request(
                method=method,
                url=f"{target_server.url}/{path}",
                headers=headers,
                data=body,
            ) as response:
                content = await response.read()

                response_headers = dict(response.headers)
                response_headers["X-Kupo-Source"] = target_server.url

                return Response(
                    content=content,
                    status_code=response.status,
                    headers=response_headers,
                )
    except Exception as e:
        logger.error("Error proxying to Kupo: %s", e)
        raise HTTPException(
            status_code=500,
            detail=f"Error proxying request: {str(e)}",
            headers={"X-Kupo-Source": target_server.url},
        ) from e
