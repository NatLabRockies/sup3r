"""sup3r forward pass CLI entry points."""

import copy
import logging
import os
from inspect import signature

import click

from sup3r import __version__
from sup3r.pipeline.forward_pass import ForwardPass, ForwardPassStrategy
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
    """Sup3r Forward Pass Command Line Interface"""
    ctx.ensure_object(dict)
    ctx.obj['VERBOSE'] = verbose


@main.command()
@click.option(
    '--config_file',
    '-c',
    required=True,
    type=click.Path(exists=True),
    help='sup3r forward pass configuration .json file.',
)
@click.option(
    '-v',
    '--verbose',
    is_flag=True,
    help='Flag to turn on debug logging. Default is not verbose.',
)
@click.pass_context
def from_config(ctx, config_file, verbose=False, pipeline_step=None):
    """Run sup3r forward pass from a config file."""

    config = BaseCLI.from_config_preflight(
        ModuleName.FORWARD_PASS, ctx, config_file, verbose
    )

    exec_kwargs = config.get('execution_control', {})
    hardware_option = exec_kwargs.pop('option', 'local')
    node_index = config.get('node_index', None)
    basename = config.get('job_name')
    log_pattern = config.get('log_pattern', None)

    sig = signature(ForwardPassStrategy)
    strategy_kwargs = {k: v for k, v in config.items() if k in sig.parameters}
    strategy = ForwardPassStrategy(**strategy_kwargs, head_node=True)

    if node_index is not None:
        nodes = (
            [node_index] if not isinstance(node_index, list) else node_index
        )
    else:
        nodes = range(len(strategy.node_chunks))
    for i_node in nodes:
        node_config = copy.deepcopy(config)
        node_config['node_index'] = i_node
        node_config['log_file'] = (
            log_pattern
            if log_pattern is None
            else os.path.normpath(log_pattern.format(node_index=i_node))
        )
        name = '{}_{}'.format(basename, str(i_node).zfill(6))
        ctx.obj['NAME'] = name
        node_config['job_name'] = name
        node_config['pipeline_step'] = pipeline_step
        cmd = ForwardPass.get_node_cmd(node_config)

        if hardware_option.lower() in AVAILABLE_HARDWARE_OPTIONS:
            kickoff_slurm_job(ctx, cmd, pipeline_step, **exec_kwargs)
        else:
            kickoff_local_job(ctx, cmd, pipeline_step)


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
        ModuleName.FORWARD_PASS,
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
    """Run sup3r forward pass locally.

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
    BaseCLI.kickoff_local_job(ModuleName.FORWARD_PASS, ctx, cmd, pipeline_step)


if __name__ == '__main__':
    try:
        main(obj={})
    except Exception:
        logger.exception('Error running sup3r forward pass CLI')
        raise
