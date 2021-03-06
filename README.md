# Fish School Search
Implementation of the [fish school search (FSS)](https://en.wikipedia.org/wiki/Fish_School_Search) proposed in the paper [**A novel algorithm based on fish school behavior**](https://ieeexplore.ieee.org/document/4811695). 

> _"An unimodal optimization algorithm inspired in the collective behavior of fish schools. Fishes swim toward the positive gradient in order to eat and gain weight. The heavier fishes are more influent in the search process, making the school move to better places in the search space"_ — From [Wikipedia](https://en.wikipedia.org/wiki/Fish_School_Search)

<p align="center">
    <img width="512" height="304" src="images/fs.gif">
</p>


## Installation

To install the dependencies, run the following command:

```bash
pip install -r requirements.txt
```

If using Conda, you can also create an environment with the requirements:

```bash
conda env create -f environment.yml
```

By default the environment name is `fish-school-search`. To activate it run:

```bash
conda activate fish-school-search
```



## Usage

Run the algorithm from the command line with:

```python
python -m fish_school_search
```

To modify any parameter of the simulation, edit `fish_school_search/__main__.py`. For more information regarding the different visualization modes check `notebooks/`.


