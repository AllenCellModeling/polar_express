# polar_express

[![Build Status](https://github.com/AllenCellModeling/polar_express/workflows/Build%20Master/badge.svg)](https://github.com/AllenCellModeling/polar_express/actions)
[![Documentation](https://github.com/AllenCellModeling/polar_express/workflows/Documentation/badge.svg)](https://AllenCellModeling.github.io/polar_express)
[![Code Coverage](https://codecov.io/gh/AllenCellModeling/polar_express/branch/master/graph/badge.svg)](https://codecov.io/gh/AllenCellModeling/polar_express)

Polar Express: 3D Image Analysis Pipeline

---

A DataStep pipeline to process 3D single-cell microscopy images, perform large-scale statistical analysis on GFP intensity distribution for target organelles, and generate summary visualizations for the data.

## Features
All steps and functionality in this package can be run as single steps or all together
by using the command line.

In general, all commands for this package will follow the format:
`polar_express {step} {command}`

* `step` is the name of the step, such as "selectdata" or "computecellmetrics"
* `command` is what you want that step to do, such as "run" or "push"

Example datasets can be accessed [here](https://open.quiltdata.com/b/allencell).

### Pipeline
To run the entire pipeline from start to finish you can simply run:

```bash
polar_express all run --dataset {path to dataset}
```

Step specific parameters can additionally be passed by simply appending them.
For example: the step `computecellmetrics` has a parameter for
`num_angular_compartments` and this can be set on both the individual step run level and
also for the entire pipeline with:

```bash
polar_express all run --dataset {path to dataset} --num_angular_compartments {integer}
```

See the [steps module in our documentation](https://allencellmodeling.github.io/polar_express/polar_express.steps.html)
for a full list of parameters for each step

#### Pipeline Config

A configuration file can be provided to the underlying `datastep` library that manages
the data storage and upload of the steps in this workflow.

The config file should simply be called `workflow_config.json` and be available from
whichever directory you run `polar_express` from. If this config is not found in the current
working directory, defaults are selected by the `datastep` package.

Here is an example of our production config:

```json
{
    "quilt_storage_bucket": "s3://allencell",
    "project_local_staging_dir": "/allen/aics/modeling/william.chen/results/polar_express"
}
```

You can even additionally attach step-specific configuration in this file by using the
name of the step like so:

```json
{
    "quilt_storage_bucket": "s3://example_config_7",
    "project_local_staging_dir": "example/config/7",
    "example": {
        "step_local_staging_dir": "example/step/local/staging/"
    }
}
```

### Individual Steps
* `polar_express selectdata run --dataset {path to dataset}`, Load 3D cell images
filtered for the target structure or organelle, or alternatively, generate artificial cells.
* `polar_express computecellmetrics run`, Compute and save key metrics relating to the
structure of the cell and GFP distribution.
* `polar_express gathertestvisualize run`, Generate visualizations for the compartmentalized
GFP distributions, including polar heatmaps, and run statistical tests over the entire dataset.

## Installation
**Install Requires:** The python package, `numpy`, must be installed prior to the
installation of this package: `pip install numpy`

**Stable Release:** `pip install polar_express`<br>
**Development Head:** `pip install git+https://github.com/AllenCellModeling/polar_express.git`

## Documentation
For full package documentation please visit
[allencellmodeling.github.io/polar_express](https://allencellmodeling.github.io/polar_express/index.html).

## Development
See [CONTRIBUTING.md](CONTRIBUTING.md) for information related to developing the code.

For more details on how this pipeline is constructed please see
[cookiecutter-stepworkflow](https://github.com/AllenCellModeling/cookiecutter-stepworkflow)
and [datastep](https://github.com/AllenCellModeling/datastep).

To add new steps to this pipeline, run `make_new_step` and follow the instructions in
[CONTRIBUTING.md](CONTRIBUTING.md)

### Developer Installation
The following two commands will install the package with dev dependencies in editable
mode and download all resources required for testing.

```bash
pip install -e .[dev]
python scripts/download_test_data.py
```

***Free software: Allen Institute Software License***

