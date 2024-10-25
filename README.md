# Setup

> Note: Conda is required for this project.

## 1. Create Conda environment if not done so

This will create a `tsp` environment, which will be the main environment for this project.

Currently installed packages/dependencies:

* `pandas`
* `numpy`
* `jupyter`

```bash
conda env create -f environment.yml
```

## 2. Activate Conda environment

On Windows:

```bash
conda activate tsp
```

On MacOS/Linux:

```bash
source activate tsp
```

## 3. Install any new dependencies if needed

```bash
conda install [NAME_OF_DEPENDENCY(IES)]
```

## 4. Update `environment.yml` with new dependencies

```bash
conda env export --no-builds > environment.yml
```

## 5. Update environment with updated `environment.yml` file

This step is for when remote changes are pulled and the working `tsp` environment already exists.

```bash
conda env update -f environment.yml
```
