# pylint: disable=all
"""Batch Job CLI entry points."""
import logging

import click
from gaps.batch import BatchJob

from sup3r import __version__

logger = logging.getLogger(__name__)


@click.group()
@click.version_option(version=__version__)
@click.option('-v', '--verbose', is_flag=True,
              help='Flag to turn on debug logging. Default is not verbose.')
@click.pass_context
def main(ctx, verbose):
    """Sup3r Batch Command Line Interface"""
    ctx.ensure_object(dict)
    ctx.obj['VERBOSE'] = verbose


@main.command()
@click.option('--config_file', '-c', required=True,
              type=click.Path(exists=True),
              help='Sup3r batch configuration json or csv file.')
@click.option('--dry-run', is_flag=True,
              help='Flag to do a dry run (make batch dirs without running).')
@click.option('--cancel', is_flag=True,
              help='Flag to cancel all jobs associated with a given pipeline.')
@click.option('--delete', is_flag=True,
              help='Flag to delete all batch job sub directories associated '
              'with the batch_jobs.csv in the current batch config directory.')
@click.option('--monitor-background', is_flag=True,
              help='Flag to monitor all batch pipelines continuously '
              'in the background using the nohup command. Note that the '
              'stdout/stderr will not be captured, but you can set a '
              'pipeline "log_file" to capture logs.')
@click.option('-v', '--verbose', is_flag=True,
              help='Flag to turn on debug logging. Default is not verbose.')
@click.pass_context
def from_config(ctx, config_file, dry_run, cancel, delete, monitor_background,
                verbose=False):
    """Run Sup3r batch from a config file."""
    ctx.ensure_object(dict)
    ctx.obj['VERBOSE'] = verbose or ctx.obj.get('VERBOSE', False)
    logger.info(
        'Starting batch job from %s (dry_run=%s, cancel=%s, delete=%s, '
        'monitor_background=%s).',
        config_file,
        dry_run,
        cancel,
        delete,
        monitor_background,
    )
    batch = BatchJob(config_file)

    if cancel:
        batch.cancel()
    elif delete:
        batch.delete()
    else:
        batch.run(dry_run=dry_run, monitor_background=monitor_background)

    logger.info('Finished batch CLI invocation for %s.', config_file)


if __name__ == '__main__':
    main(obj={})
