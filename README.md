# Exoplanet ML

Machine learning models and utilities for exoplanet science.

## Code Author

Chris Shallue: [@cshallue](https://github.com/cshallue)

## Walkthrough

You can jump straight to the [AstroNet walkthrough](exoplanet-ml/astronet/README.md#walkthrough).

Otherwise, click through to the desired directory as outlined below.

## Directories

[astronet/](exoplanet-ml/astronet/)

* A neural network for identifying exoplanets in light curves. Contains code for:
  * Downloading and preprocessing Kepler light curves.
  * Building different types of neural network classification models.
  * Training and evaluating a new model.
  * Using a trained model to generate new predictions.

[astrowavenet/](exoplanet-ml/astrowavenet/)

* A generative model for light curves.

[light_curve/](exoplanet-ml/light_curve)

* Utilities for operating on light curves. These include:
  * Reading Kepler data from `.fits` files.
  * Applying a median filter to smooth and normalize a light curve.
  * Phase folding, splitting, removing periodic events, etc.
* [light_curve/fast_ops/](exoplanet-ml/light_curve/fast_ops) contains optimized
C++ light curve operations.

[tf_util/](exoplanet-ml/tf_util)

* Shared TensorFlow utilities.

[third_party/](exoplanet-ml/third_party/)

* Utilities derived from third party code.


# Setup

## Required Packages

* **TensorFlow** ([instructions](https://www.tensorflow.org/install/))
* **Pandas** ([instructions](http://pandas.pydata.org/pandas-docs/stable/install.html))
* **NumPy** ([instructions](https://docs.scipy.org/doc/numpy/user/install.html))
* **SciPy** ([instructions](https://scipy.org/install.html))
* **AstroPy** ([instructions](http://www.astropy.org/))
* **PyDl** ([instructions](https://pypi.python.org/pypi/pydl))
* **Bazel** ([instructions](https://docs.bazel.build/versions/master/install.html))
* **Abseil Python Common Libraries** ([instructions](https://github.com/abseil/abseil-py))

## Run Unit Tests

Verify that all dependencies are satisfied by running the unit tests:

```bash
cd exoplanet-ml  # Bazel must run from a directory with a WORKSPACE file
bazel test astronet/... astrowavenet/... light_curve/... tf_util/... third_party/...
```

# Citation

If you find this code useful, please cite our paper:

Shallue, C. J., & Vanderburg, A. (2018). Identifying Exoplanets with Deep
Learning: A Five-planet Resonant Chain around Kepler-80 and an Eighth Planet
around Kepler-90. *The Astronomical Journal*, 155(2), 94.

Full text available at [*The Astronomical Journal*](http://iopscience.iop.org/article/10.3847/1538-3881/aa9e09/meta).

# Disclaimer

This is not an official Google product.

# Exoplanet ML

Machine learning models and utilities for exoplanet science.

## Code Author

Chris Shallue: [@cshallue](https://github.com/cshallue)

## Walkthrough

You can jump straight to the [AstroNet walkthrough](exoplanet-ml/astronet/README.md#walkthrough).

Otherwise, click through to the desired directory as outlined below.

## Directories

[astronet/](exoplanet-ml/astronet/)

* A neural network for identifying exoplanets in light curves. Contains code for:
  * Downloading and preprocessing Kepler light curves.
  * Building different types of neural network classification models.
  * Training and evaluating a new model.
  * Using a trained model to generate new predictions.

[astrowavenet/](exoplanet-ml/astrowavenet/)

* A generative model for light curves.

[light_curve/](exoplanet-ml/light_curve)

* Utilities for operating on light curves. These include:
  * Reading Kepler data from `.fits` files.
  * Applying a median filter to smooth and normalize a light curve.
  * Phase folding, splitting, removing periodic events, etc.
* [light_curve/fast_ops/](exoplanet-ml/light_curve/fast_ops) contains optimized
C++ light curve operations.

[tf_util/](exoplanet-ml/tf_util)

* Shared TensorFlow utilities.

[third_party/](exoplanet-ml/third_party/)

* Utilities derived from third party code.

# Setup

## Required Packages

* **TensorFlow** ([instructions](https://www.tensorflow.org/install/))
* **Pandas** ([instructions](http://pandas.pydata.org/pandas-docs/stable/install.html))
* **NumPy** ([instructions](https://docs.scipy.org/doc/numpy/user/install.html))
* **SciPy** ([instructions](https://scipy.org/install.html))
* **AstroPy** ([instructions](http://www.astropy.org/))
* **PyDl** ([instructions](https://pypi.python.org/pypi/pydl))
* **Bazel** ([instructions](https://docs.bazel.build/versions/master/install.html))
* **Abseil Python Common Libraries** ([instructions](https://github.com/abseil/abseil-py))

> **Optional:** If you prefer, you can run the project using Docker, which already contains all dependencies and the pretrained model.

~~~bash
docker build -t exoplanet-ml .
docker run -it exoplanet-ml /bin/bash
~~~

Once inside the container, you can run the model directly.

---

### Run Unit Tests

Verify that all dependencies are satisfied by running the unit tests:

~~~bash
cd exoplanet-ml  # Bazel must run from a directory with a WORKSPACE file
bazel test astronet/... astrowavenet/... light_curve/... tf_util/... third_party/...
~~~

---

### Using the Model

#### For Kepler data

~~~bash
python kepler_input.py {kepler_id}
~~~

#### For TESS data

~~~bash
python tess_input.py {tess_id}
~~~

> Replace `{kepler_id}` or `{tess_id}` with the corresponding ID.

#### Example

~~~bash
python kepler_input.py 010797460
~~~

Output:

~~~text
Prediction: 0.981166
~~~

This prints the probability that the provided light curve contains an exoplanet.

---

### Citation

If you find this code useful, please cite our paper:

Shallue, C. J., & Vanderburg, A. (2018). Identifying Exoplanets with Deep Learning: A Five-planet Resonant Chain around Kepler-80 and an Eighth Planet around Kepler-90. *The Astronomical Journal*, 155(2), 94.

Full text available at [*The Astronomical Journal*](http://iopscience.iop.org/article/10.3847/1538-3881/aa9e09/meta).

---

### Disclaimer

This is not an official Google product.
