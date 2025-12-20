import typer
import uvicorn

app = typer.Typer()


@app.command()
def serve(
    host: str = typer.Option("localhost", help="Host to run the server on"),
    port: int = typer.Option(1442, help="Port to run the server on"),
    workers: int = typer.Option(1, help="Number of worker processes"),
):
    """
    Start the FastAPI server using Uvicorn.
    """
    uvicorn.run("kupo.server:app", host=host, port=port, workers=workers)


if __name__ == "__main__":
    app()
