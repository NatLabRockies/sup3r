"""Sup3r command line interface (CLI)."""

import logging

import click
from gaps import Pipeline

from ._version import __version__
from .batch.batch_cli import from_config as batch_cli
from .bias.bias_calc_cli import from_config as bias_calc_cli
from .pipeline.forward_pass_cli import from_config as fwp_cli
from .pipeline.pipeline_cli import from_config as pipe_cli
from .postprocessing.data_collect_cli import from_config as dc_cli
from .qa.qa_cli import from_config as qa_cli
from .solar.solar_cli import from_config as solar_cli
from .utilities import ModuleName

logger = logging.getLogger(__name__)


@click.group()
@click.version_option(version=__version__)
@click.option(
    '--config_file',
    '-c',
    required=False,
    type=click.Path(exists=True),
    help='sup3r configuration file json for a single module.',
)
@click.option(
    '-v',
    '--verbose',
    is_flag=True,
    help='Flag to turn on debug logging. Default is not verbose.',
)
@click.pass_context
def main(ctx, config_file, verbose):
    """Sup3r command line interface.

    Try using the following commands to pull up the help pages for the
    respective sup3r CLIs::

        $ sup3r --help

        $ sup3r -c config.json pipeline --help

        $ sup3r -c config.json forward-pass --help

        $ sup3r -c config.json data-collect --help

    Typically, a good place to start is to set up a sup3r job with a pipeline
    config that points to several sup3r modules that you want to run in serial.
    You would call the sup3r pipeline CLI using either of these equivalent
    commands::

        $ sup3r -c config_pipeline.json pipeline

        $ sup3r-pipeline from-config -c config_pipeline.json

    See the help pages of the module CLIs for more details on the config files
    for each CLI.
    """
    ctx.ensure_object(dict)
    ctx.obj['CONFIG_FILE'] = config_file
    ctx.obj['VERBOSE'] = verbose


@main.command()
@click.option(
    '-v', '--verbose', is_flag=True, help='Flag to turn on debug logging.'
)
@click.pass_context
def forward_pass(ctx, verbose):
    """Sup3r forward pass to super-resolve data.

    The sup3r forward pass is where all the magic happens. This module takes in
    low resolution data from source files (e.g. WRF or GCM outputs), loads a
    saved Sup3rGan model, does a forward pass of the low res data through the
    generator, and then saves the output high resolution data to disk. This
    module also handles multi-node and multi-core parallelization.

    You can call the forward-pass module via the sup3r-pipeline CLI, or call it
    directly with either of these equivalent commands::

        $ sup3r -c config_fwp.json forward-pass

        $ sup3r-forward-pass from-config -c config_fwp.json

    A sup3r forward pass config.json file can contain any arguments or keyword
    arguments required to initialize the
    :class:`sup3r.pipeline.strategy.ForwardPassStrategy` module. The config
    also has several optional arguments: ``log_pattern``, ``log_level``, and
    ``execution_control``. Here's a small example forward pass config::

        \b
        {
            "file_paths": "./source_files*.nc",
            "model_kwargs": {
                "model_dir": "./sup3r_model/"
            },
            "out_pattern": "./output/sup3r_out_{file_id}.h5",
            "log_pattern": "./logs/log_{node_index}.log",
            "log_level": "DEBUG",
            "fwp_chunk_shape": [5, 5, 3],
            "spatial_pad": 1,
            "temporal_pad": 1,
            "max_nodes": 1,
            "output_workers": 1,
            "pass_workers": 8,
            "input_handler_kwargs": {"max_workers": 1},
            "execution_control": {
                "option": "kestrel",
                "walltime": 4,
                "alloc": "sup3r"
            }
        }

    Note that the ``execution_control`` block contains kwargs that would
    be required to distribute the job on multiple nodes on the NLR HPC.
    To run the job locally, use ``execution_control: {"option": "local"}``.
    """  # noqa : D301
    config_file = ctx.obj['CONFIG_FILE']
    verbose = any((verbose, ctx.obj['VERBOSE']))
    ctx.invoke(fwp_cli, config_file=config_file, verbose=verbose)


@main.command()
@click.option(
    '-v', '--verbose', is_flag=True, help='Flag to turn on debug logging.'
)
@click.pass_context
def solar(ctx, verbose):
    """Sup3r solar module to convert GAN output clearsky ratio to irradiance

    Typically we train solar GAN's on clearsky ratio to remove the dependence
    on known variables like solar position and clearsky irradiance. This module
    converts the clearsky ratio output by the GAN in the forward-pass step to
    actual irradiance values (ghi, dni, and dhi). This should be run after the
    forward-pass but before the data-collect step.

    You can call the solar module via the sup3r-pipeline CLI, or call it
    directly with either of these equivalent commands::

        $ sup3r -c config_solar.json solar

        $ sup3r-solar from-config -c config_solar.json

    A sup3r solar config.json file can contain any arguments or keyword
    arguments required to call the
    :meth:`sup3r.solar.solar.Solar.run_temporal_chunks` method. You do not need
    to include the ``i_t_chunk`` input, this is added by the CLI. The config
    also has several optional arguments: ``log_pattern``, ``log_level``, and
    ``execution_control``. Here's a small example solar config::

        \b
        {
            "fp_pattern": "./chunks/sup3r*.h5",
            "nsrdb_fp": "/datasets/NSRDB/current/nsrdb_2015.h5",
            "max_nodes": 100,
            "execution_control": {
                "option": "kestrel",
                "walltime": 4,
                "alloc": "sup3r"
            }
        }

    Note that the ``execution_control`` block contains kwargs that would
    be required to distribute the job on multiple nodes on the NLR HPC.
    To run the job locally, use ``execution_control: {"option": "local"}``.
    """  # noqa : D301
    config_file = ctx.obj['CONFIG_FILE']
    verbose = any((verbose, ctx.obj['VERBOSE']))
    ctx.invoke(solar_cli, config_file=config_file, verbose=verbose)


@main.command()
@click.option(
    '-v', '--verbose', is_flag=True, help='Flag to turn on debug logging.'
)
@click.pass_context
def bias_calc(ctx, verbose):
    """Sup3r bias correction calculation module to create bias correction
    factors for low res input data.

    This is typically used to understand and correct the biases in global
    climate model (GCM) data from CMIP. Bias correcting GCM data is a very
    common process when dealing with climate data and this helps automate that
    process prior to passing GCM data through sup3r generative models. This is
    typically the first step in a GCM downscaling pipeline.

    You can call the bias calc module via the sup3r-pipeline CLI, or call it
    directly with either of these equivalent commands::

        $ sup3r -c config_bias.json bias-calc

        $ sup3r-bias-calc from-config -c config_bias.json

    A sup3r bias calc config.json file can contain any arguments or keyword
    arguments required to call the

    The config has high level ``bias_calc_class`` and ``jobs`` keys. The
    ``bias_calc_class`` is a class name from the :mod:`sup3r.bias`
    module, and the ``jobs`` argument is a list of kwargs required to
    initialize the ``bias_calc_class`` and run the ``bias_calc_class.run()``
    method (for example, see
    :meth:`sup3r.bias.bias_calc.LinearCorrection.run`). There are also has
    several optional arguments: ``log_pattern``, ``log_level``, and
    ``execution_control``. Here's a small example bias calc config::

        \b
        {
            "bias_calc_class": "LinearCorrection",
            "jobs": [
                {
                    "base_fps" : ["/datasets/WIND/HRRR/HRRR_2015.h5"],
                    "bias_fps": ["./ta_day_EC-Earth3-Veg_ssp585.nc"],
                    "base_dset": "u_100m",
                    "bias_feature": "u_100m",
                    "target": [20, -130],
                    "shape": [48, 95]
                 }
            ],
            "execution_control": {
                "option": "kestrel",
                "walltime": 4,
                "alloc": "sup3r"
            }
        }

    Note that the ``execution_control`` block contains kwargs that would
    be required to distribute the job on multiple nodes on the NLR HPC.
    To run the job locally, use ``execution_control: {"option": "local"}``.
    """  # noqa : D301
    config_file = ctx.obj['CONFIG_FILE']
    verbose = any((verbose, ctx.obj['VERBOSE']))
    ctx.invoke(bias_calc_cli, config_file=config_file, verbose=verbose)


@main.command()
@click.option(
    '-v', '--verbose', is_flag=True, help='Flag to turn on debug logging.'
)
@click.pass_context
def data_collect(ctx, verbose):
    """Sup3r data collection following forward pass.

    The sup3r data-collect module can be used to collect time-chunked files
    that were spit out by multi-node forward pass jobs. You can call the
    data-collect module via the sup3r-pipeline CLI, or call it directly with
    either of these equivalent commands::

        $ sup3r -c config_collect.json data-collect

        $ sup3r-data-collect from-config -c config_collect.json

    A sup3r data-collect config.json file can contain any arguments or keyword
    arguments required to run the
    :meth:`sup3r.postprocessing.collectors.base.BaseCollector.collect`
    method. The config also has several optional arguments: ``log_file``,
    ``log_level``,
    and ``execution_control``. Here's a small example data-collect config::

        \b
        {
            "file_paths": "./outputs/*.h5",
            "out_file": "./outputs/output_file.h5",
            "features": ["windspeed_100m", "winddirection_100m"],
            "log_file": "./logs/collect.log",
            "execution_control": {"option": "local"},
            "log_level": "DEBUG"
        }

    Note that the ``execution_control`` has the same options as forward-pass
    and you can set ``"option": "kestrel"`` to run on the NLR HPC.
    """  # noqa : D301
    config_file = ctx.obj['CONFIG_FILE']
    verbose = any((verbose, ctx.obj['VERBOSE']))
    ctx.invoke(dc_cli, config_file=config_file, verbose=verbose)


@main.command()
@click.option(
    '-v', '--verbose', is_flag=True, help='Flag to turn on debug logging.'
)
@click.pass_context
def qa(ctx, verbose):
    """Sup3r QA module following forward pass and collection.

    The sup3r QA module can be used to verify how well the high-resolution
    sup3r resolved outputs adhere to the low-resolution source data. You can
    call the QA module via the sup3r-pipeline CLI, or call it
    directly with either of these equivalent commands::

        $ sup3r -c config_qa.json qa

        $ sup3r-qa from-config -c config_qa.json

    A sup3r QA config.json file can contain any arguments or keyword
    arguments required to initialize the :class:`sup3r.qa.qa.Sup3rQa` class.
    The config also has several optional arguments: ``log_file``,
    ``log_level``, and ``execution_control``. Here's a small example
    QA config::

        \b
        {
            "source_file_paths": "./source_files*.nc",
            "out_file_path": "./outputs/collected_output_file.h5",
            "s_enhance": 25,
            "t_enhance": 24,
            "temporal_coarsening_method": "average",
            "log_file": "./logs/qa.log",
            "execution_control": {"option": "local"},
            "log_level": "DEBUG"
        }

    Note that the ``execution_control`` has the same options as forward-pass
    and you can set ``"option": "kestrel"`` to run on the NLR HPC.
    """  # noqa : D301
    config_file = ctx.obj['CONFIG_FILE']
    verbose = any((verbose, ctx.obj['VERBOSE']))
    ctx.invoke(qa_cli, config_file=config_file, verbose=verbose)


@main.group(invoke_without_command=True)
@click.option(
    '--cancel',
    is_flag=True,
    help='Flag to cancel all jobs associated with a given pipeline.',
)
@click.option(
    '--monitor',
    is_flag=True,
    help='Flag to monitor pipeline jobs continuously. '
    'Default is not to monitor (kick off jobs and exit).',
)
@click.option(
    '--background',
    is_flag=True,
    help='Flag to monitor pipeline jobs continuously '
    'in the background using the nohup command. Note that the '
    'stdout/stderr will not be captured, but you can set a '
    'pipeline "log_file" to capture logs.',
)
@click.option(
    '-v', '--verbose', is_flag=True, help='Flag to turn on debug logging.'
)
@click.pass_context
def pipeline(ctx, cancel, monitor, background, verbose):
    """Execute multiple steps in a Sup3r pipeline.

    Typically, a good place to start is to set up a sup3r job with a pipeline
    config that points to several sup3r modules that you want to run in serial.
    You would call the sup3r pipeline CLI using either of these equivalent
    commands::

        $ sup3r -c config_pipeline.json pipeline

        $ sup3r-pipeline from-config -c config_pipeline.json

    A typical sup3r pipeline config.json file might look like this::

        \b
        {
            "logging": {"log_level": "DEBUG"},
            "pipeline": [
                {"forward-pass": "./config_fwp.json"},
                {"data-collect": "./config_collect.json"}
            ]
        }

    See the other CLI help pages for what the respective module configs
    require.
    """  # noqa: D301
    if ctx.invoked_subcommand is None:
        config_file = ctx.obj['CONFIG_FILE']
        verbose = any((verbose, ctx.obj['VERBOSE']))
        ctx.invoke(
            pipe_cli,
            config_file=config_file,
            cancel=cancel,
            monitor=monitor,
            background=background,
            verbose=verbose,
        )


@main.group(invoke_without_command=True)
@click.option(
    '--dry-run',
    is_flag=True,
    help='Flag to do a dry run (make batch dirs without running).',
)
@click.option(
    '--cancel',
    is_flag=True,
    help='Flag to cancel all jobs associated with a given batch.',
)
@click.option(
    '--delete',
    is_flag=True,
    help='Flag to delete all batch job sub directories associated '
    'with the batch_jobs.csv in the current batch config directory.',
)
@click.option(
    '--monitor-background',
    is_flag=True,
    help='Flag to monitor all batch pipelines continuously '
    'in the background using the nohup command. Note that the '
    'stdout/stderr will not be captured, but you can set a '
    'pipeline "log_file" to capture logs.',
)
@click.option(
    '-v', '--verbose', is_flag=True, help='Flag to turn on debug logging.'
)
@click.pass_context
def batch(ctx, dry_run, cancel, delete, monitor_background, verbose):
    """Create and run multiple sup3r project directories based on batch
    permutation logic.

    The sup3r batch module (built on the reV batch functionality) is a way to
    create and run many sup3r pipeline projects based on permutations of
    key-value pairs in the run config files. A user configures the batch file
    by creating one or more "sets" that contain one or more arguments (keys
    found in config files) that are to be parameterized. For example, in the
    config below, four sup3r pipelines will be created where arg1 and arg2 are
    set to [0, "a"], [0, "b"], [1, "a"], [1, "b"] in config_fwp.json::

        \b
        {
            "pipeline_config": "./config_pipeline.json",
            "sets": [
              {
                "args": {
                  "arg1": [0, 1],
                  "arg2": ["a", "b"],
                },
                "files": ["./config_fwp.json"],
                "set_tag": "set1"
              }
        }

    Note that you can use multiple "sets" to isolate parameter permutations.
    """  # noqa : D301
    if ctx.invoked_subcommand is None:
        config_file = ctx.obj['CONFIG_FILE']
        verbose = any((verbose, ctx.obj['VERBOSE']))
        ctx.invoke(
            batch_cli,
            config_file=config_file,
            dry_run=dry_run,
            cancel=cancel,
            delete=delete,
            monitor_background=monitor_background,
            verbose=verbose,
        )


Pipeline.COMMANDS[ModuleName.FORWARD_PASS] = fwp_cli
Pipeline.COMMANDS[ModuleName.SOLAR] = solar_cli
Pipeline.COMMANDS[ModuleName.DATA_COLLECT] = dc_cli
Pipeline.COMMANDS[ModuleName.QA] = qa_cli
Pipeline.COMMANDS[ModuleName.BIAS_CALC] = bias_calc_cli


if __name__ == '__main__':
    try:
        main(obj={})
    except Exception:
        logger.exception('Error running sup3r CLI')
        raise
