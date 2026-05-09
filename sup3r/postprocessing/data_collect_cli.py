"""sup3r data collection CLI entry points."""

import copy
import logging

import click

from sup3r import __version__
from sup3r.postprocessing.collectors import CollectorH5, CollectorNC
from sup3r.preprocessing.utilities import get_source_type
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
    """Sup3r Data Collection Command Line Interface"""
    ctx.ensure_object(dict)
    ctx.obj['VERBOSE'] = verbose


@main.command()
@click.option(
    '--config_file',
    '-c',
    required=True,
    type=click.Path(exists=True),
    help='sup3r data collection configuration json file.',
)
@click.option(
    '-v',
    '--verbose',
    is_flag=True,
    help='Flag to turn on debug logging. Default is not verbose.',
)
@click.pass_context
def from_config(ctx, config_file, verbose=False, pipeline_step=None):
    """Run sup3r data collection from a config file. If dset_split is True this
    each feature will be collected into a separate file."""
    config = BaseCLI.from_config_preflight(
        ModuleName.DATA_COLLECT, ctx, config_file, verbose
    )

    dset_split = config.get('dset_split', False)
    exec_kwargs = config.get('execution_control', {})
    hardware_option = exec_kwargs.pop('option', 'local')
    source_type = get_source_type(config['file_paths'])
    collector_types = {'h5': CollectorH5, 'nc': CollectorNC}
    Collector = collector_types[source_type]

    logger.info(
        'Preparing data collection from %s to %s using %s output files.',
        config_file,
        config['out_file'],
        source_type,
    )

    configs = [config]
    if dset_split:
        configs = []
        for feature in config['features']:
            f_config = copy.deepcopy(config)
            f_out_file = config['out_file'].replace(
                f'.{source_type}', f'_{feature}.{source_type}'
            )
            f_job_name = config['job_name'] + f'_{feature}'
            f_log_file = config.get('log_file', None)
            if f_log_file is not None:
                f_log_file = f_log_file.replace('.log', f'_{feature}.log')
            f_config.update({
                'features': [feature],
                'out_file': f_out_file,
                'job_name': f_job_name,
                'log_file': f_log_file,
            })

            configs.append(f_config)

    for config in configs:
        ctx.obj['NAME'] = config['job_name']
        config['pipeline_step'] = pipeline_step
        cmd = Collector.get_node_cmd(config)

        logger.info(
            'Queueing data collection job "%s" for %s features.',
            config['job_name'],
            len(config['features']),
        )

        if hardware_option.lower() in AVAILABLE_HARDWARE_OPTIONS:
            kickoff_slurm_job(ctx, cmd, pipeline_step, **exec_kwargs)
        else:
            kickoff_local_job(ctx, cmd, pipeline_step)

    logger.info(
        'Finished queueing data collection work for %s jobs.', len(configs)
    )


def kickoff_local_job(ctx, cmd, pipeline_step=None):
    """Run sup3r data collection locally.

    Parameters
    ----------
    ctx : click.pass_context
        Click context object where ctx.obj is a dictionary
    cmd : str
        Command to be submitted in shell script. Example:
            'python -m sup3r.cli data_collect -c <config_file>'
    pipeline_step : str, optional
        Name of the pipeline step being run. If ``None``, the
        ``pipeline_step`` will be set to the ``module_name``,
        mimicking old reV behavior. By default, ``None``.
    """
    BaseCLI.kickoff_local_job(ModuleName.DATA_COLLECT, ctx, cmd, pipeline_step)


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
            'python -m sup3r.cli data-collect -c <config_file>'
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
        ModuleName.DATA_COLLECT,
        ctx,
        cmd,
        alloc,
        memory,
        walltime,
        feature,
        stdout_path,
        pipeline_step,
    )


if __name__ == '__main__':
    try:
        main(obj={})
    except Exception:
        logger.exception('Error running sup3r data collection CLI')
        raise
