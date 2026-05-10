"""Sup3r base CLI class."""
import logging
import os

import click
from gaps import Status
from gaps.config import load_config
from rex.utilities.execution import SubprocessManager
from rex.utilities.hpc import SLURM
from rex.utilities.loggers import init_mult

from . import ModuleName
from .utilities import safe_serialize

logger = logging.getLogger(__name__)
AVAILABLE_HARDWARE_OPTIONS = ('kestrel', 'eagle', 'slurm')


class SlurmManager(SLURM):
    """GAPs-compliant SLURM manager"""

    def check_status_using_job_id(self, job_id):
        """Check the status of a job using the HPC queue and job ID.

        Parameters
        ----------
        job_id : int
            Job integer ID number.

        Returns
        -------
        status : str | None
            Queue job status string or `None` if not found.
        """
        return self.check_status(job_id=job_id)


class BaseCLI:
    """Base CLI class used to create CLI for modules in ModuleName"""

    @classmethod
    def from_config(cls, module_name, module_class, ctx, config_file, verbose,
                    pipeline_step=None):
        """Run sup3r module from a config file.


        Parameters
        ----------
        module_name : str
            Module name string from :class:`sup3r.utilities.ModuleName`.
        module_class : Object
            Class object used to call get_node_cmd(config).
            e.g. Sup3rQa.get_node_cmd(config)
        ctx : click.pass_context
            Click context object where ctx.obj is a dictionary
        config_file : str
            Path to config file provided all needed inputs to module_class
        verbose : bool
            Whether to run in verbose mode.
        pipeline_step : str, optional
            Name of the pipeline step being run. If ``None``, the
            ``pipeline_step`` will be set to the ``module_name``,
            mimicking old reV behavior. By default, ``None``.
        """
        config = cls.from_config_preflight(
            module_name, ctx, config_file, verbose
        )
        config['pipeline_step'] = pipeline_step

        exec_kwargs = config.get('execution_control', {})
        hardware_option = exec_kwargs.pop('option', 'local')

        logger.info(
            'Preparing sup3r %s from %s with hardware=%s.',
            module_name,
            config_file,
            hardware_option,
        )

        cmd = module_class.get_node_cmd(config)

        if hardware_option.lower() in AVAILABLE_HARDWARE_OPTIONS:
            cls.kickoff_slurm_job(module_name, ctx, cmd,
                                  pipeline_step=pipeline_step,
                                  **exec_kwargs)
        else:
            cls.kickoff_local_job(module_name, ctx, cmd,
                                  pipeline_step=pipeline_step)

        logger.info(
            'Finished sup3r %s submission for %s.',
            module_name,
            config_file,
        )

    @classmethod
    def from_config_preflight(cls, module_name, ctx, config_file, verbose):
        """Parse conifg file prior to running sup3r module.

        Parameters
        ----------
        module_name : str
            Module name string from :class:`sup3r.utilities.ModuleName`.
        module_class : Object
            Class object used to call get_node_cmd(config).
            e.g. Sup3rQa.get_node_cmd(config)
        ctx : click.pass_context
            Click context object where ctx.obj is a dictionary
        config_file : str
            Path to config file provided all needed inputs to module_class
        verbose : bool
            Whether to run in verbose mode.

        Returns
        -------
        config : dict
            Dictionary corresponding to config_file
        """
        cls.check_module_name(module_name)

        ctx.ensure_object(dict)
        ctx.obj['VERBOSE'] = verbose
        status_dir = os.path.dirname(os.path.abspath(config_file))
        ctx.obj['OUT_DIR'] = status_dir
        config = load_config(config_file)
        config['status_dir'] = status_dir
        log_file = config.get('log_file', None)
        log_pattern = config.get('log_pattern', None)
        config_verbose = config.get('log_level', 'INFO')
        config_verbose = config_verbose == 'DEBUG'
        verbose = any([verbose, config_verbose, ctx.obj['VERBOSE']])
        exec_kwargs = config.get('execution_control', {})
        hardware_option = exec_kwargs.get('option', 'local')

        log_dir = log_file or log_pattern
        log_dir = log_dir if log_dir is None else os.path.dirname(log_dir)

        init_mult(
            f'sup3r_{module_name.replace("-", "_")}',
            log_dir,
            modules=[__name__, 'sup3r'],
            verbose=verbose,
        )

        if log_pattern is not None:
            os.makedirs(os.path.dirname(log_pattern), exist_ok=True)
            if '.log' not in log_pattern:
                log_pattern += '.log'
            if '{node_index}' not in log_pattern:
                log_pattern = log_pattern.replace('.log', '_{node_index}.log')

        exec_kwargs['stdout_path'] = os.path.join(status_dir, 'stdout/')
        logger.debug('Found execution kwargs: {}'.format(exec_kwargs))
        logger.debug('Hardware run option: "{}"'.format(hardware_option))

        name = f'sup3r_{module_name.replace("-", "_")}'
        name += '_{}'.format(os.path.basename(status_dir))
        job_name = config.get('job_name', None)
        if job_name is not None:
            name = job_name
        ctx.obj['NAME'] = name
        config['job_name'] = name
        config['status_dir'] = status_dir

        return config

    @classmethod
    def check_module_name(cls, module_name):
        """Make sure module_name is a valid member of the ModuleName class"""
        msg = (
            'Module name must be in ModuleName class. Received '
            f'{module_name}.'
        )
        assert module_name in ModuleName, msg

    @classmethod
    def kickoff_slurm_job(
        cls,
        module_name,
        ctx,
        cmd,
        alloc='sup3r',
        memory=None,
        walltime=4,
        feature=None,
        stdout_path='./stdout/',
        pipeline_step=None,
    ):
        """Run sup3r module on HPC via SLURM job submission.

        Parameters
        ----------
        module_name : str
            Module name string from :class:`sup3r.utilities.ModuleName`.
        ctx : click.pass_context
            Click context object where ctx.obj is a dictionary
        cmd : str
            Command to be submitted in SLURM shell script. Example:
                'python -m sup3r.cli <module_name> -c <config_file>'
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
        pipeline_step : str, optional
            Name of the pipeline step being run. If ``None``, the
            ``pipeline_step`` will be set to the ``module_name``,
            mimicking old reV behavior. By default, ``None``.
        """
        cls.check_module_name(module_name)
        if pipeline_step is None:
            pipeline_step = module_name

        name = ctx.obj['NAME']
        out_dir = ctx.obj['OUT_DIR']
        slurm_manager = ctx.obj.get('SLURM_MANAGER', None)
        if slurm_manager is None:
            slurm_manager = SlurmManager()
            ctx.obj['SLURM_MANAGER'] = slurm_manager

        status = Status.retrieve_job_status(
            out_dir,
            pipeline_step=pipeline_step,
            job_name=name,
            subprocess_manager=slurm_manager,
        )
        job_failed = 'fail' in str(status).lower()
        job_submitted = status != 'not submitted'

        msg = f'sup3r {module_name} CLI failed to submit jobs!'
        if status == 'successful':
            msg = (
                f'Job "{name}" is successful in status json found in '
                f'"{out_dir}", not re-running.'
            )
        elif not job_failed and job_submitted and status is not None:
            msg = (
                f'Job "{name}" was found with status "{status}", not '
                'resubmitting'
            )
        else:
            job_info = f"{module_name}"
            if pipeline_step != module_name:
                job_info = f"{job_info} (pipeline step {pipeline_step!r})"
            logger.info(
                f'Running sup3r {job_info} on SLURM with node '
                f'name "{name}".'
            )
            out = slurm_manager.sbatch(
                cmd,
                alloc=alloc,
                memory=memory,
                walltime=walltime,
                feature=feature,
                name=name,
                stdout_path=stdout_path,
            )[0]
            if out:
                msg = (
                    f'Kicked off sup3r {job_info} job "{name}" '
                    f'(SLURM jobid #{out}).'
                )

            # add job to sup3r status file.
            Status.mark_job_as_submitted(
                out_dir,
                pipeline_step=pipeline_step,
                job_name=name,
                replace=True,
                job_attrs={'job_id': out, 'hardware': 'kestrel'},
            )

        click.echo(msg)
        logger.info(msg)

    @classmethod
    def kickoff_local_job(cls, module_name, ctx, cmd, pipeline_step=None):
        """Run sup3r module locally.

        Parameters
        ----------
        module_name : str
            Module name string from :class:`sup3r.utilities.ModuleName`.
        ctx : click.pass_context
            Click context object where ctx.obj is a dictionary
        cmd : str
            Command to be submitted in shell script. Example:
                'python -m sup3r.cli <module_name> -c <config_file>'
        pipeline_step : str, optional
            Name of the pipeline step being run. If ``None``, the
            ``pipeline_step`` will be set to the ``module_name``,
            mimicking old reV behavior. By default, ``None``.
        """
        cls.check_module_name(module_name)
        if pipeline_step is None:
            pipeline_step = module_name

        name = ctx.obj['NAME']
        out_dir = ctx.obj['OUT_DIR']
        subprocess_manager = SubprocessManager

        status = Status.retrieve_job_status(
            out_dir, pipeline_step=pipeline_step, job_name=name
        )
        job_failed = 'fail' in str(status).lower()
        job_submitted = status != 'not submitted'

        msg = f'sup3r {module_name} CLI failed to submit jobs!'
        if status == 'successful':
            msg = (
                f'Job "{name}" is successful in status json found in '
                f'"{out_dir}", not re-running.'
            )
        elif not job_failed and job_submitted and status is not None:
            msg = (
                f'Job "{name}" was found with status "{status}", not '
                'resubmitting'
            )
        else:
            job_info = f"{module_name}"
            if pipeline_step != module_name:
                job_info = f"{job_info} (pipeline step {pipeline_step!r})"
            logger.info(
                f'Running sup3r {job_info} locally with job '
                f'name "{name}".'
            )
            Status.mark_job_as_submitted(
                out_dir,
                pipeline_step=pipeline_step,
                job_name=name,
                replace=True
            )
            subprocess_manager.submit(cmd)
            msg = f'Completed sup3r {job_info} job "{name}".'

        click.echo(msg)
        logger.info(msg)

    @classmethod
    def add_status_cmd(cls, config, pipeline_step, cmd):
        """Append status file command to command for executing given module

        Parameters
        ----------
        config : dict
            sup3r config with all necessary args and kwargs to run given
            module.
        pipeline_step : str
            Name of the pipeline step being run.
        cmd : str
            String including command to execute given module.

        Returns
        -------
        cmd : str
            Command string with status file command included if job_name is
            not None
        """
        job_name = config.get('job_name', None)
        status_dir = config.get('status_dir', None)
        if job_name is not None and status_dir is not None:
            status_file_arg_str = f'"{status_dir}", '
            status_file_arg_str += f'pipeline_step="{pipeline_step}", '
            status_file_arg_str += f'job_name="{job_name}", '
            status_file_arg_str += 'attrs=job_attrs'

            cmd += 'job_attrs = {};\n'.format(
                safe_serialize(config)
                .replace("null", "None")
                .replace("false", "False")
                .replace("true", "True")
            )
            cmd += 'job_attrs.update({"job_status": "successful"});\n'
            cmd += 'job_attrs.update({"time": t_elap});\n'
            cmd += f"Status.make_single_job_file({status_file_arg_str})"

        cmd_log = '\n\t'.join(cmd.split('\n'))
        logger.debug(f'Running command:\n\t{cmd_log}')

        return cmd
