from pathlib import Path
from typing import Annotated

import typer

from orchestration import run_pipeline
from report_output import render_console, save_report

app = typer.Typer(add_completion=False)


@app.command()
def main(
    query: Annotated[str, typer.Argument(help="The research question to investigate.")],
    output: Annotated[
        Path, typer.Option("--output", help="Directory to save the .md/.json report to.")
    ] = Path("reports"),
    verbose: Annotated[
        bool, typer.Option("--verbose", help="Show all agent iterations.")
    ] = False,
) -> None:
    report = run_pipeline(query, verbose=verbose)

    md_path, json_path = save_report(report, output)
    print(f"[Report] Saved to {md_path.parent}/{md_path.stem}.{{md,json}}")
    print()
    print(render_console(report))


if __name__ == "__main__":
    app()
