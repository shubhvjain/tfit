import os
import click
from typing import Dict, Any

from tfitpy.config import load_config_file
from tfitpy.setup import ensure_all_data


@click.group()
@click.option(
    "--config",
    "config_path",
    type=str,  # Changed: no click.Path to allow ~/$VAR expansion
    envvar="TFIT_CONFIG",
    help="Path to tfit JSON config file.",
)
@click.pass_context
def cli(ctx: click.Context, config_path: str | None):
    """
    TFit: Assessing the combinatorial potential of
    Transcription Factors in Gene Regulation
    """
    # Load config if provided
    if config_path:
        config_path = os.path.expanduser(os.path.expandvars(config_path))
        try:
            ctx.obj = {"config": load_config_file(config_path)}
            click.echo(f"Loaded config: {config_path}")
        except Exception as e:
            click.echo(f"Config error: {e}", err=True)
            ctx.obj = {"config": {}}
    else:
        ctx.obj = {"config": {}}
        # click.echo("Using default config")


@cli.command()
@click.option("--output", "-o", default="~/.config/tfit/config.json", help="Output config file path")
@click.pass_context
def init(ctx: click.Context, output: str):
    """Generate blank config file."""
    from tfitpy.config import save_blank_config
    
    path = save_blank_config(output)
    click.echo(f"Blank config created: {path}")
    click.echo("Edit it and use with --config or TFIT_CONFIG env var.")


@cli.command()
@click.pass_context
def setup(ctx: click.Context):
    """Download all required biological databases to make the package ready to be run"""
    cfg: Dict[str, Any] = ctx.obj["config"]
    ensure_all_data(cfg)