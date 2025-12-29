"""CLI entry point for kupo-py server mode."""

import typer
import uvicorn

app = typer.Typer(help="kupo-py: A Python client and server for Kupo.")


@app.callback(invoke_without_command=True)
def main(ctx: typer.Context):
    """kupo-py: A Python client and server for Kupo."""
    if ctx.invoked_subcommand is None:
        # Show help if no subcommand is provided
        print(ctx.get_help())
        raise typer.Exit(0)


@app.command()
def serve(
    host: str = typer.Option("localhost", help="Host address to bind to"),
    port: int = typer.Option(1442, help="Port number to listen on"),
    workers: int = typer.Option(1, help="Number of Uvicorn worker processes"),
):
    """Start the kupo-py load balancing server.

    The server proxies requests to multiple Kupo instances configured via
    KUPO_PY_URL_* environment variables, with optional DBSync fallback.
    """
    uvicorn.run("kupo.server:app", host=host, port=port, workers=workers)


if __name__ == "__main__":
    app()
