"""sup3r solar CLI entry points."""

import copy
import logging
import os

import click
import numpy as np

from sup3r import __version__
from sup3r.solar import Solar
from sup3r.utilities import ModuleName
from sup3r.utilities.cli import AVAILABLE_HARDWARE_OPTIONS, BaseCLI

logger = logging.getLogger(__name__)


@click.group()
@click.version_option(version=__version__)
@click.option(
    '-v',
    '--verbose',
    is_flag=True,
    help='Flag to turn on debug logging. Default is not verbose.',
)
@click.pass_context
def main(ctx, verbose):
    """Sup3r Solar Command Line Interface"""
    ctx.ensure_object(dict)
    ctx.obj['VERBOSE'] = verbose


@main.command()
@click.option(
    '--config_file',
    '-c',
    required=True,
    type=click.Path(exists=True),
    help='sup3r solar configuration .json file.',
)
@click.option(
    '-v',
    '--verbose',
    is_flag=True,
    help='Flag to turn on debug logging. Default is not verbose.',
)
@click.pass_context
def from_config(ctx, config_file, verbose=False, pipeline_step=None):
    """Run sup3r solar from a config file."""
    config = BaseCLI.from_config_preflight(
        ModuleName.SOLAR, ctx, config_file, verbose
    )
    exec_kwargs = config.get('execution_control', {})
    hardware_option = exec_kwargs.pop('option', 'local')
    log_pattern = config.get('log_pattern', None)
    fp_pattern = config['fp_pattern']
    basename = config['job_name']
    fp_sets, _, temporal_ids, _, _ = Solar.get_sup3r_fps(fp_pattern)
    temporal_ids = sorted(set(temporal_ids))
    max_nodes = config.get('max_nodes', len(temporal_ids))
    max_nodes = min((max_nodes, len(temporal_ids)))
    logger.info(
        'Solar module found %s sets of chunked source files to run on. '
        'Submitting to %s nodes based on the number of temporal chunks %s '
        'and the requested number of nodes %s',
        len(fp_sets),
        max_nodes,
        len(temporal_ids),
        config.get('max_nodes', None),
    )

    temporal_id_chunks = np.array_split(temporal_ids, max_nodes)
    for i_node, temporal_ids in enumerate(temporal_id_chunks):
        node_config = copy.deepcopy(config)
        node_config['log_file'] = (
            log_pattern
            if log_pattern is None
            else os.path.normpath(log_pattern.format(node_index=i_node))
        )
        name = '{}_{}'.format(basename, str(i_node).zfill(6))
        ctx.obj['NAME'] = name
        node_config['job_name'] = name
        node_config['pipeline_step'] = pipeline_step

        node_config['temporal_ids'] = list(temporal_ids)
        cmd = Solar.get_node_cmd(node_config)

        logger.info(
            'Queueing solar node %s as job "%s".',
            i_node,
            name,
        )

        if hardware_option.lower() in AVAILABLE_HARDWARE_OPTIONS:
            kickoff_slurm_job(ctx, cmd, pipeline_step, **exec_kwargs)
        else:
            kickoff_local_job(ctx, cmd, pipeline_step)

    logger.info('Finished queueing solar work for %s nodes.', max_nodes)


def kickoff_slurm_job(
    ctx,
    cmd,
    pipeline_step=None,
    alloc='sup3r',
    memory=None,
    walltime=4,
    feature=None,
    stdout_path='./stdout/',
):
    """Run sup3r on HPC via SLURM job submission.

    Parameters
    ----------
    ctx : click.pass_context
        Click context object where ctx.obj is a dictionary
    cmd : str
        Command to be submitted in SLURM shell script. Example:
            'python -m sup3r.cli forward_pass -c <config_file>'
    pipeline_step : str, optional
        Name of the pipeline step being run. If ``None``, the
        ``pipeline_step`` will be set to the ``module_name``,
        mimicking old reV behavior. By default, ``None``.
    alloc : str
        HPC project (allocation) handle. Example: 'sup3r'.
    memory : int
        Node memory request in GB.
    walltime : float
        Node walltime request in hours.
    feature : str
        Additional flags for SLURM job. Format is "--qos=high"
        or "--depend=[state:job_id]". Default is None.
    stdout_path : str
        Path to print .stdout and .stderr files.
    """
    BaseCLI.kickoff_slurm_job(
        ModuleName.SOLAR,
        ctx,
        cmd,
        alloc,
        memory,
        walltime,
        feature,
        stdout_path,
        pipeline_step,
    )


def kickoff_local_job(ctx, cmd, pipeline_step=None):
    """Run sup3r solar locally.

    Parameters
    ----------
    ctx : click.pass_context
        Click context object where ctx.obj is a dictionary
    cmd : str
        Command to be submitted in shell script. Example:
            'python -m sup3r.cli forward_pass -c <config_file>'
    pipeline_step : str, optional
        Name of the pipeline step being run. If ``None``, the
        ``pipeline_step`` will be set to the ``module_name``,
        mimicking old reV behavior. By default, ``None``.
    """
    BaseCLI.kickoff_local_job(ModuleName.SOLAR, ctx, cmd, pipeline_step)


if __name__ == '__main__':
    try:
        main(obj={})
    except Exception:
        logger.exception('Error running sup3r solar CLI')
        raise
