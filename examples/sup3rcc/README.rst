################
Sup3rCC Examples
################

Super-Resolution for Renewable Energy Resource Data with Climate Change Impacts (Sup3rCC) is one application of the sup3r software. In this work, we train generative models to create high-resolution (4km hourly) wind, solar, and temperature data based on coarse (100km daily) global climate model data (GCM). The generative models and high-resolution output data are publicly available via the `Open Energy Data Initiative (OEDI) <https://data.openei.org/submissions/5839>`__ and via HSDS at the bucket ``nrel-pds-hsds`` and HSDS path ``/nrel/sup3rcc/``. This set of examples lays out basic ways to use the Sup3rCC models and data.

Sup3rCC Data Access
--------------------

For high level details on accessing the NLR renewable energy resource datasets including Sup3rCC, see the `rex docs pages <https://natlabrockies.github.io/rex/misc/examples.nrel_data.html>`__

The Sup3rCC data and models are publicly available in a public AWS S3 bucket. The data files and models can be downloaded directly from there to your local machine or an EC2 instance using the `OEDI data explorer <https://data.openei.org/s3_viewer?bucket-nrel-pds-sup3rcc>`__ or the `AWS CLI <https://aws.amazon.com/cli/>`__. A word of caution: there's a lot of data here. The smallest Sup3rCC file for just a single variable is ~20 GB, and a full year of data is ~200 GB.

The Sup3rCC data is also loaded into `HSDS <https://www.hdfgroup.org/solutions/highly-scalable-data-service-hsds/>`__ so that you may stream the data via the `NLR developer API <https://developer.nlr.gov/signup/>`__ or your own HSDS server. This is the best option if you're not going to want the full annual dataset over the whole United States. See these `rex instructions <https://natlabrockies.github.io/rex/misc/examples.hsds.html>`__ for more details on how to access this data with HSDS and rex.

The data can now be opened remotely with ``xarray`` and ``rex``. See the docs `here <https://natlabrockies.github.io/rex/misc/examples.xarray.html>`__ for instructions.

Directory Structure
-------------------

The Sup3rCC directory contains downscaled data for multiple projections of future climate change. For example, a file from the initial data release ``sup3rcc_conus_ecearth3_ssp585_r1i1p1f1_wind_2015.h5`` is downscaled from the climate model "MRI-ESM-2.0" for climate change scenario SSP5 8.5 and variant label r1i1p1f1. The file contains wind variables for the year 2015. Note that this will represent the climate from 2015, but not the actual "real-world" weather we experienced.

Within the S3 bucket there is also a folder ``models`` providing pre-trained Sup3rCC generative machine learning models.

Example Sup3rCC Data Usage
--------------------------

The jupyter notebook in this example shows some basic code to access and explore the data. You can walk through the `example notebook <https://github.com/NatLabRockies/sup3r/tree/main/examples/sup3rcc/using_the_data.ipynb>`__. You can clone this repo, setup a basic python environment with `rex <https://github.com/NatLabRockies/rex>`__, and run the notebook on your own.

Running Sup3rCC Models
----------------------

In a first-of-a-kind data product, we have released the pre-trained Sup3rCC generative machine learning models along with the sup3r software so that anyone working at the intersection of energy and climate may create their own high-resolution renewable energy resource data from GCM input. You might want to do this if you have your own GCMs or climate scenarios that you're interested in studying.

To run the Sup3rCC models, follow these instructions:

#. Decide what kind of hardware you're going to use. You could technically run Sup3rCC on a desktop computer, but you will need lots of CPU cores and RAM (we use compute nodes with ~100 cores and >170 GB of RAM). We recommend a high-performance-computing cluster if you have access to one, or an `AWS Parallel Cluster <https://aws.amazon.com/hpc/parallelcluster/>`__ if you do not.
#. Download the Sup3rCC models to your hardware using the AWS CLI: ``$ aws s3 cp s3://nrel-pds-sup3rcc/models/``
#. Download the GCM data that you want to downscale from `CMIP6 <https://esgf-node.llnl.gov/search/cmip6/>`__
#. Setup the Sup3rCC software. We recommend using `miniconda <https://docs.conda.io/en/latest/miniconda.html>`__ to manage your python environments. You can create a sup3r environment with the conda file in this example directory: ``$ conda env create -n sup3rcc --file env.yml``
#. Copy this examples directory to your hardware. You're going to be using the folder structure in ``/sup3r/examples/sup3rcc/run_configs`` as your project directories (``/sup3r/`` is a git clone of the sup3r software repo).
#. Navigate to ``/sup3r/examples/sup3rcc/run_configs/nearsurf/`` and update all of the filepaths in the config files for the source GCM data, Sup3rCC models, and exogenous data sources (e.g. the ``nsrdb_clearsky.h5`` file downloaded from OEDI).
#. Update the execution control parameters in the ``config_fwp.json`` file based on the hardware you're running on.
#. You can either run ``sup3r-batch`` to setup multiple run years, or ``sup3r-pipeline`` to run just one job. We recommend starting with ``sup3r-pipeline`` (more on the sup3r `CLI <https://natlabrockies.github.io/sup3r/_cli/sup3r.html>`__).
#. To run ``sup3r-pipeline``, make sure you are in the directory with the ``config_pipeline.json`` and ``config_fwp.json`` files, and then run this command: ``python -m sup3r.cli -c config_pipeline.json pipeline``
#. If you're running on a slurm cluster, this will kick off a number of jobs that you can see with the ``squeue`` command. If you're running locally, your terminal should now be running the Sup3rCC models. The software will create a ``./logs/`` directory in which you can monitor the progress of your jobs.
#. The ``sup3r-pipeline`` is designed to run several modules in serial, with each module running multiple chunks in parallel. Once the first module (forward-pass) finishes, you'll want to run ``python -m sup3r.cli -c config_pipeline.json pipeline`` again. This will clean up status files and kick off the next step in the pipeline (if the current step was successful).

Note that you can get significantly better performance by pre-loading the variable-by-variable and multi-year CMIP6 files using the `Sup3rCC data handler <https://natlabrockies.github.io/sup3r/_autosummary/sup3r.preprocessing.data_handlers.nc_cc.DataHandlerNCforCC.html#sup3r.preprocessing.data_handlers.nc_cc.DataHandlerNCforCC>`__ and saving single files per year with all necessary variables for use in the generative runs.

Nuances of Sup3rCC
------------------

The Sup3rCC dataset is quite unlike the legacy NLR historical wind and solar datasets. As such, we expect there will be some confusion about how to use the data. There are some nuances of the data enumerated below. If you have any questions about how to apply the Sup3rCC data to your work, please reach out to Grant Buster (Grant.Buster@nlr.gov).

#. Sup3rCC data is based on global climate model (GCM) data, which does not represent historical weather, only historical climate. So for example, Sup3rCC 2015 does not represent the actual historical weather in 2015, just the historical climate in 2015.
#. Sup3rCC data represents just one possible future climate subject to deep uncertainties. Do not use the Sup3rCC data as an accurate prediction of future weather. Some uncertanties about our future climate can be quantified by exploring a large ensemble of GCM data across multiple climate scenarios and multiple models.
#. Sup3rCC cannot represent many meteorological events that are not skillfully represented in GCM data (e.g., hurricanes, tornadoes, mesoscale convective storms, wildfires, etc…).
#. Sup3rCC does not currently use land use data and only understands phenomena like urban heat islands via the bias correction of the GCM data with historical reanalysis data. Application of Sup3rCC to individual cities may benefit from statistical validation using historical ground measurement data.
#. For v0.2.2, the Sup3rCC data was double bias-corrected at the 100-km daily and 4-km hourly resolutions using baseline historical data from ERA5, Daymet, the NSRDB, HRRR, and WTK. This improves upon the original v0.1.0 Sup3rCC release which only did the low-resolution bias correction and had substantial residual bias in the high-res outputs. The new v0.2.2 data should have minimal bias over the 2000-2019 historical period and the Sup3rCC data should be usable out-of-the-box without further bias correction. See the Sup3rCC v0.2.2 manuscript for more details.

Sup3rCC Versions
----------------

The Sup3rCC data has versions that coincide with the sup3r software versions. Note that not every sup3r software version will have a corresponding Sup3rCC data release, but every Sup3rCC data release will have a corresponding sup3r software version. This table records versions of Sup3rCC data releases. Sup3rCC generative models may have slightly different versions than the data. The version in the Sup3rCC .h5 file attribute can be inspected to verify the actual version of the data you are using.

.. list-table::
    :widths: auto
    :header-rows: 1

    * - Version
      - Effective Date
      - Notes
    * - 0.1.0
      - 6/27/2023
      - Initial Sup3rCC release with two GCMs and one climate scenario. Known issues: few years used for bias correction, simplistic GCM bias correction method, mean bias in high-res output especially in wind and solar data, imperfect wind diurnal cycles when compared to WTK and timing of diurnal peak temperature when compared to observation.
    * - 0.2.2
      - 3/24/2025
      - Release of the second-generation Sup3rCC data with six GCMs across two climate scenarios (5x SSP2-4.5 and 1x SSP5-8.5). This version includes new generative models that have a larger effective receptive field for improved spatiotemporal weather dynamics over large-areas and improved diurnal shapes. This version also includes seasonal double-bias correction with Quantile Delta Mapping (QDM) at the 100km and 4km resolutions over a longer historical period (20-40 years) for greatly reduced historical climate bias. This release includes "sup3rcc_models_202412" that can be used with sup3r software v0.2.2 and phygnn v0.0.30 to reproduce this data release, and "sup3rcc_models_202507" that can be used with sup3r v0.2.3 and phygnn v0.0.31 with some additional non-ML performance improvements.

Recommended Citation
--------------------

Buster, G. et al. Second-Generation Downscaled Earth System Model Data using Generative Machine Learning. Data in Brief (2025) doi:10.1016/j.dib.2025.111774.

Buster, G., Benton, B.N., Glaws, A. et al. High-resolution meteorology with climate change impacts from global climate model data using generative machine learning. Nature Energy (2024). https://doi.org/10.1038/s41560-024-01507-9

Buster, Grant, Benton, Brandon, Glaws, Andrew, & King, Ryan. Super-Resolution for Renewable Energy Resource Data with Climate Change Impacts (Sup3rCC). United States. https://dx.doi.org/10.25984/1970814. https://data.openei.org/submissions/5839.


Acknowledgements
----------------

This work was authored by the National Laboratory of the Rockies for the U.S. Department of Energy (DOE) under Contract No. DE-AC36-08GO28308. This research was supported by the Grid Modernization Initiative of the U.S. Department of Energy (DOE) as part of its Grid Modernization Laboratory Consortium, a strategic partnership between DOE and the national laboratories to bring together leading experts, technologies, and resources to collaborate on the goal of modernizing the nation’s grid. Funding provided by the DOE Office of Energy Efficiency and Renewable Energy (EERE), the DOE Office of Electricity (OE), DOE Grid Deployment Office (GDO), the DOE Advanced Scientific Computing Research (ASCR) program, the DOE Solar Energy Technologies Office (SETO), and the Laboratory Directed Research and Development (LDRD) program at the National Lab of the Rockies. The research was performed using computational resources sponsored by the DOE Office of Energy Efficiency and Renewable Energy and located at the National Lab of the Rockies. The views expressed in the article do not necessarily represent the views of the DOE or the U.S. Government. The U.S. Government retains and the publisher, by accepting the article for publication, acknowledges that the U.S. Government retains a nonexclusive, paid-up, irrevocable, worldwide license to publish or reproduce the published form of this work, or allow others to do so, for U.S. Government purposes.
