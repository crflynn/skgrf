skgrf
=====

|actions| |travis| |rtd| |pypi| |pyversions|

.. |actions| image:: https://github.com/crflynn/skgrf/workflows/build/badge.svg
    :target: https://github.com/crflynn/skgrf/actions

.. |travis| image:: https://img.shields.io/travis/crflynn/skgrf-wheels/main.svg?logo=travis&label=wheels
    :target: https://travis-ci.org/crflynn/skgrf-wheels

.. |rtd| image:: https://img.shields.io/readthedocs/skgrf.svg
    :target: http://skgrf.readthedocs.io/en/latest/

.. |pypi| image:: https://img.shields.io/pypi/v/skgrf.svg
    :target: https://pypi.python.org/pypi/skgrf

.. |pyversions| image:: https://img.shields.io/pypi/pyversions/skgrf.svg
    :target: https://pypi.python.org/pypi/skgrf

``skgrf`` provides `scikit-learn <https://scikit-learn.org/stable/index.html>`__ compatible Python bindings to the C++ random forest implementation, `grf <https://github.com/grf-labs/grf>`__, using `Cython <https://cython.readthedocs.io/en/latest/>`__.

The latest release of ``skgrf`` uses version `1.2.0 <https://github.com/grf-labs/grf/releases/tag/v1.2.0>`__ of ``grf``.

``skgrf`` is still in development. Please create issues for any discrepancies or errors. PRs welcome.


Installation
------------

``skgrf`` is available on `pypi <https://pypi.org/project/skgrf>`__ and can be installed via pip:

.. code-block:: bash

    pip install skgrf

Estimators
----------

* GRFBoostedRegressor
* GRFCausalRegressor
* GRFInstrumentalRegressor
* GRFLocalLinearRegressor
* GRFQuantileRegressor
* GRFRegressor
* GRFSurvival

Usage
-----

GRFRegressor
~~~~~~~~~~~~

The ``GRFRegressor`` predictor uses ``grf``'s RegressionPredictionStrategy class.

.. code-block:: python

    from sklearn.datasets import load_boston
    from sklearn.model_selection import train_test_split
    from skgrf.ensemble import GRFRegressor
    
    X, y = load_boston(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    
    rfr = GRFRegressor()
    rfr.fit(X_train, y_train)
    
    predictions = rfr.predict(X_test)
    print(predictions)
    # [31.81349144 32.2734354  16.51560285 11.90284392 39.69744341 21.30367911
    #  19.52732937 15.82126562 26.49528961 11.27220097 16.02447197 20.01224404
    #  ...
    #  20.70674263 17.09041289 12.89671205 20.79787926 21.18317924 25.45553279
    #  20.82455595]


GRFQuantileRegressor
~~~~~~~~~~~~~~~~~~~~

The ``GRFQuantileRegressor`` predictor uses ``grf``'s QuantilePredictionStrategy class.

.. code-block:: python

    from sklearn.datasets import load_boston
    from sklearn.model_selection import train_test_split
    from skgrf.ensemble import GRFQuantileRegressor
    
    X, y = load_boston(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    
    gqr = GRFQuantileRegressor(quantiles=[0.1, 0.9])
    gqr.fit(X_train, y_train)
    
    predictions = gqr.predict(X_test)
    print(predictions)
    # [[21.9 50. ]
    # [ 8.5 24.5]
    # ...
    # [ 8.4 18.6]
    # [ 8.1 20. ]]

License
-------

``skgrf`` is licensed under `GPLv3 <https://github.com/crflynn/skgrf/blob/main/LICENSE.txt>`__.

Development
-----------

To develop locally, it is recommended to have ``asdf``, ``make`` and a C++ compiler already installed. After cloning, run ``make setup``. This will setup the grf submodule, install python and poetry from ``.tool-versions``, install dependencies using poetry, copy the grf source code into skgrf, and then build and install skgrf in the local virtualenv.

To format code, run ``make fmt``. This will run isort and black against the .py files.

To run tests and inspect coverage, run ``make test``.

To rebuild in place after making changes, run ``make build``.

To create python package artifacts, run ``make dist``.

To build and view documentation, run ``make docs``.
