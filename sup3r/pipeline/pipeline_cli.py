# pylint: disable=all
"""Pipeline CLI entry points."""
import logging

import click
from gaps.cli.pipeline import pipeline

from sup3r import __version__

logger = logging.getLogger(__name__)


@click.group()
@click.version_option(version=__version__)
@click.option('-v', '--verbose', is_flag=True,
              help='Flag to turn on debug logging. Default is not verbose.')
@click.pass_context
def main(ctx, verbose):
    """Sup3r Pipeline Command Line Interface"""
    ctx.ensure_object(dict)
    ctx.obj['VERBOSE'] = verbose


@main.command()
@click.option('--config_file', '-c', required=True,
              type=click.Path(exists=True),
              help='sup3r pipeline configuration json file.')
@click.option('--cancel', is_flag=True,
              help='Flag to cancel all jobs associated with a given pipeline.')
@click.option('--monitor', is_flag=True,
              help='Flag to monitor pipeline jobs continuously. '
              'Default is not to monitor (kick off jobs and exit).')
@click.option('--background', is_flag=True,
              help='Flag to monitor pipeline jobs continuously in the '
              'background using the nohup command. This only works with the '
              '--monitor flag. Note that the stdout/stderr will not be '
              'captured, but you can set a pipeline log_file to capture logs.')
@click.option('-v', '--verbose', is_flag=True,
              help='Flag to turn on debug logging. Default is not verbose.')
@click.pass_context
def from_config(ctx, config_file, cancel, monitor, background, verbose=False):
    """Run sup3r pipeline from a config file."""
    ctx.ensure_object(dict)
    ctx.obj['VERBOSE'] = verbose or ctx.obj.get('VERBOSE', False)
    logger.info(
        'Starting pipeline from %s (cancel=%s, monitor=%s, background=%s).',
        config_file,
        cancel,
        monitor,
        background,
    )
    pipeline(config_file, cancel, monitor, background)
    logger.info('Finished pipeline invocation for %s.', config_file)


if __name__ == '__main__':
    try:
        main(obj={})
    except Exception:
        logger.exception('Error running sup3r pipeline CLI')
        raise
