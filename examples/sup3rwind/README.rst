###################
Sup3rWind Examples
###################

Super-Resolution for Renewable Energy Resource Data with Wind from Reanalysis Data (Sup3rWind) is one application of the sup3r software. In this work, we train generative models to create high-resolution (2km 5-minute) wind data based on coarse (30km hourly) ERA5 data. The generative models, high-resolution output data, and training data is publicly available via the `Open Energy Data Initiative (OEDI) <https://data.openei.org/s3_viewer?bucket=nrel-pds-wtk&prefix=sup3rwind%2F>`__ and via HSDS at the bucket ``nrel-pds-hsds`` and path ``/nrel/wtk/sup3rwind``. This data covers recent historical time periods for an expanding selection of countries.

Sup3rWind Data Access
----------------------

The Sup3rWind data and models are publicly available in a public AWS S3 bucket. The data files can be downloaded directly from there to your local machine or an EC2 instance using the `OEDI data explorer <https://data.openei.org/s3_viewer?bucket=nrel-pds-wtk&prefix=sup3rwind%2F>`__ or the `AWS CLI <https://aws.amazon.com/cli/>`__. A word of caution: there's a lot of data here. The smallest Sup3rWind file for just a single variable at 2-km 5-minute resolution is 130 GB.

The Sup3rWind data is also loaded into `HSDS <https://www.hdfgroup.org/solutions/highly-scalable-data-service-hsds/>`__ so that you may stream the data via the `NREL developer API <https://developer.nrel.gov/signup/>`__ or your own HSDS server. This is the best option if you're not going to want a full annual dataset. See these `rex instructions <https://nrel.github.io/rex/misc/examples.hsds.html>`__ for more details on how to access this data with HSDS and rex.

Sup3rWind Data Usage
---------------------

Sup3rWind data can be used in generally the same way as `Sup3rCC <https://nrel.github.io/sup3r/examples/sup3rcc.html>`__ data, with the condition that Sup3rWind includes only wind data and ancillary variables for modeling wind energy generation. Refer to the Sup3rCC `example notebook <https://github.com/NREL/sup3r/tree/main/examples/sup3rcc/using_the_data.ipynb>`__ for usage patterns.

Running Sup3rWind Models
-------------------------

The process for running the Sup3rWind models is much the same as for `Sup3rCC <https://nrel.github.io/sup3r/examples/sup3rcc.html>`__.

#. Download the Sup3rWind models to your hardware using the AWS CLI: ``$ aws s3 cp s3://nrel-pds-wtk/sup3rwind/models/``
#. Setup the Sup3rWind software. We recommend using `miniconda <https://docs.conda.io/en/latest/miniconda.html>`__ to manage your python environments. You can create a sup3r environment with the conda file in this example directory: ``$ conda env create -n sup3rwind --file env.yml``
#. Download the ERA5 data that you want to downscale from `ERA5-single-levels <https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels?tab=overview/>`__ and/or `ERA5-pressure-levels <https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-pressure-levels?tab=overview/>`__ using the `EraDownloader <https://nrel.github.io/sup3r/_autosummary/sup3r.utilities.era_downloader.html#module-sup3r.utilities.era_downloader/>`__.
#. Copy this examples directory to your hardware. You're going to be using the folder structure in ``/sup3r/examples/sup3rwind/run_configs`` as your project directories (``/sup3r/`` is a git clone of the sup3r software repo).
#. Navigate to ``/sup3r/examples/sup3rwind/run_configs/wind/`` and/or ``sup3r/examples/sup3rwind/run_configs/trhp`` and update all of the filepaths in the config files for the source ERA5 data, Sup3rWind models, and exogenous data sources (e.g. the ``topography`` source file).
#. Update the execution control parameters in the ``config_fwp_spatial.json`` file based on the hardware you're running on.
#. Run ``sup3r-pipeline`` to run just one job. There are also batch options for running multiple jobs, but we recommend starting with ``sup3r-pipeline`` (more on the sup3r `CLI <https://nrel.github.io/sup3r/_cli/sup3r.html>`__).
#. To run ``sup3r-pipeline``, make sure you are in the directory with the ``config_pipeline.json`` and ``config_fwp_spatial.json`` files, and then run this command: ``python -m sup3r.cli -c config_pipeline.json pipeline``
#. If you're running on a slurm cluster, this will kick off a number of jobs that you can see with the ``squeue`` command. If you're running locally, your terminal should now be running the Sup3rWind models. The software will create a ``./logs/`` directory in which you can monitor the progress of your jobs.
#. The ``sup3r-pipeline`` is designed to run several modules in serial, with each module running multiple chunks in parallel. Once the first module (forward-pass) finishes, you'll want to run ``python -m sup3r.cli -c config_pipeline.json pipeline`` again. This will clean up status files and kick off the next step in the pipeline (if the current step was successful).

You can also checkout the `example notebook <https://github.com/NREL/sup3r/tree/main/examples/sup3rwind/running_sup3r_models.ipynb>`__ for how to run models without config files.

Training from scratch
---------------------

To train Sup3rWind models from scratch use the public training `data <https://data.openei.org/s3_viewer?bucket=nrel-pds-wtk&prefix=sup3rwind%2Ftraining_data%2F>`__. This data is for training the spatial enhancement models only. The 2024-01 `models <https://data.openei.org/s3_viewer?bucket=nrel-pds-wtk&prefix=sup3rwind%2Fmodels%2Fsup3rwind_models_202401%2F>`__ perform spatial enhancement in two steps, 3x from ERA5 to coarsened WTK and 5x from coarsened WTK to uncoarsened WTK. The currently used approach performs spatial enhancement in a single 15x step.

For a given year and training domain, initialize low-resolution and high-resolution data handlers and wrap these in a dual rasterizer object. Do this for as many years and training regions as desired, and use these containers to initialize a batch handler. To train models for 3x spatial enhancement use ``hr_spatial_coarsen=5`` in the ``hr_dh``. To train models for 15x (the currently used approach) ``hr_spatial_coarsen=1``. (Refer to tests and docs for information on additional arguments, denoted by the ellipses)::

  from sup3r.preprocessing import DataHandler, DualBatchHandler, DualRasterizer
  containers = []
  for tdir in training_dirs:
    lr_dh = DataHandler(f"{tdir}/lr_*.h5", ...)
    hr_dh = DataHandler(f"{tdir}/hr_*.h5", hr_spatial_coarsen=...)
    container = DualRasterizer({'low_res': lr_dh, 'high_res': hr_dh}, ...)
    containers.append(container)
  bh = DualBatchHandler(train_containers=containers, ...)

To train a 5x model use the ``hr_*.h5`` files for both the ``lr_dh`` and the ``hr_dh``. Use ``hr_spatial_coarsen=3`` in the ``lr_dh`` and ``hr_spatial_coarsen=1`` in the ``hr_dh``::

  for tdir in training_dirs:
    lr_dh = DataHandler(f"{tdir}/hr_*.h5", hr_spatial_coarsen=3, ...)
    hr_dh = DataHandler(f"{tdir}/hr_*.h5", hr_spatial_coarsen=1, ...)
    container = DualRasterizer({'low_res': lr_dh, 'high_res': hr_dh}, ...)
    containers.append(container)
  bh = DualBatchHandler(train_containers=containers, ...)


Initialize a 3x, 5x, or 15x spatial enhancement model, with 14 output channels, and train for the desired number of epochs. (The 3x and 5x generator configs can be copied from the ``model_params.json`` files in each OEDI model `directory <https://data.openei.org/s3_viewer?bucket=nrel-pds-wtk&prefix=sup3rwind%2Fmodels%2Fsup3rwind_models_202401%2F>`__. The 15x generator config can be created from the OEDI model configs by changing the spatial enhancement factor or from the configs in the repo by changing the enhancement factor and the number of output channels)::

  from sup3r.models import Sup3rGan
  model = Sup3rGan(gen_layers="./gen_config.json", disc_layers="./disc_config.json", ...)
  model.train(batch_handler, ...)


Sup3rWind Versions
-------------------

The Sup3rWind data has versions that coincide with the sup3r software versions. Note that not every sup3r software version will have a corresponding Sup3rWind data release, but every Sup3rWind data release will have a corresponding sup3r software version.

.. list-table::
    :widths: auto
    :header-rows: 1

    * - Version
      - Effective Date
      - Notes
    * - 0.1.2
      - 3/15/2024
      - Initial release of Sup3rWind for Ukraine, Moldova, and part of Romania. Includes 2-km 5-minute wind speed and wind direction data and 2-km hourly wind speed, wind direction, pressure, temperature, and relative humidity data for 2000-2023.
    * - 0.2.4
      - 12/15/2025
      - Initial release of Sup3rWind for South America. Includes 2-km 5-minute wind speed and wind direction data and 2-km hourly wind speed, wind direction, pressure, temperature, and relative humidity data for 2005-2024.


Recommended Citation
---------------------

Benton, B. N., Buster, G., Pinchuk, P., Glaws, A., King, R. N., Maclaurin, G., & Chernyakhovskiy, I. Super-Resolution for Renewable Energy Resource Data with Wind from Reanalysis Data and Application to Ukraine, Energies (2025) https://doi.org/10.3390/en18143769.

Acknowledgements
-----------------

This work was authored by the National Renewable Energy Laboratory, for the U.S. Department of Energy (DOE) under Contract No. DE-AC36-08GO28308. Funding provided by the DOE Grid Deployment Office (GDO), the DOE Advanced Scientific Computing Research (ASCR) program, the DOE Solar Energy Technologies Office (SETO), and the Laboratory Directed Research and Development (LDRD) program at the National Renewable Energy Laboratory. The research was performed using computational resources sponsored by the DOE Office of Energy Efficiency and Renewable Energy and located at the National Renewable Energy Laboratory. The views expressed in the article do not necessarily represent the views of the DOE or the U.S. Government. The U.S. Government retains and the publisher, by accepting the article for publication, acknowledges that the U.S. Government retains a nonexclusive, paid-up, irrevocable, worldwide license to publish or reproduce the published form of this work, or allow others to do so, for U.S. Government purposes.
