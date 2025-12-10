import click

import tfit.setup as s

@click.group()
def cli():
    """TFit :  Assessing the combinatorial potential of Transcription Factors in Gene Regulation"""
    pass

@cli.command()
def setup():
    """Download all required biological databases"""
    s.ensure_all_data()
    click.echo(f"Setup complete! Data stored at: {s.DATA_DIR}")

if __name__ == "__main__":
    cli()
