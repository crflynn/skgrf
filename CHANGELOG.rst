Release Changelog
-----------------

0.3.0 (2021-12-09)
~~~~~~~~~~~~~~~~~~

* Allow X to contain NaN values on all estimators except local linear.
* Allow ``skgrf`` predictors to work with shap using ``skgrf.utils.shap.shap_patch`` context manager
* Fix Tree.values for classifiers
* Fix Tree.feature to use proper value for leaf nodes
* Fix package includes to prevent installing extra files to site-packages
* Drop support for Python 3.6

0.2.1 (2021-09-03)
~~~~~~~~~~~~~~~~~~

* Use oldest supported numpy for builds

0.2.0 (2021-08-23)
~~~~~~~~~~~~~~~~~~

* Pin sklearn version to >=0.23.0,<1.0
* Changed tree details calculations to be optional due to expensive calculations
* Added more documentation around tree detail interface

0.1.1 (2021-07-20)
~~~~~~~~~~~~~~~~~~

* Fix to include grf src (accidentally removed in 0.1.0)

0.1.0 (2021-07-04)
~~~~~~~~~~~~~~~~~~

* Change the default number of estimators in causal regression forest from 100 to 2000
* Add validation to ensure the resulting records after taking sample fraction is always at least 1
* Change ``n_features_`` attribute to ``n_features_in_`` for consistency with sklearn
* Add validation to ``sample_weight``. Because of internal rng, this ensures that passing no weights is equivalent to passing a vector of ones
* Add validation to the number of features in the test set.
* Fitted estimators now hold a pointer to a C++ forest, eliminating the need for deserialization on predict which makes predictions faster
* Change the attribute name from ``grf_forest_`` to ``boosted_forests_`` on ``GRFBoostedRegressor``
* Add decision tree regressors for each forest type (except boosted)
* Add a ``GRFBaseTree`` class for shared methods on decision trees
* Add ``estimators_`` property to forests which return a list of tree estimators
* Add ``get_estimator`` method to forests to extract a single estimator
* Fix a bug on local linear forest where ``ll_split_variables_`` was not being set properly
* Add ``get_feature_importances`` to forests
* Add ``get_split_frequencies`` to forests
* Add ``get_kernel_weights`` to forests
* Add ``COPYING``, ``README.md``, ``REFERENCE.md`` source files from grf to package source files
* Remove leading underscore from all ``grf_forest_`` dictionary keys
* Implement most of the sklearn low-level ``Tree`` API into decision trees
* Add ``criterion`` (str) attribute to all estimators
* Remove arguments related to sparse data from bindings
* Update ``grf`` to 2.0, and update bindings
* Add ``GRFClassifier`` and ``GRFTreeClassifier`` estimators
* Change ``classes_`` and ``n_classes_`` to ``clusters_`` and ``n_clusters_``.  ``classes_`` and ``n_classes_`` are now used for target classes in classification estimators
* Remove the ``DataNumpy.cpp`` class. This is no longer needed with the ``grf`` 2.0 updates to the ``Data.cpp`` class
* Rename the ``DataNumpy`` Cython class to ``Data``
* Remove extern cpp includes from Cython ``.pxd`` file and add ``cpp`` files to ``Extension``'s ``source`` arg
* Rename forest estimators to ``GRFForest...``

0.0.1 (2021-04-11)
~~~~~~~~~~~~~~~~~~

* Modify serialize_forest function to support linux wheels.

0.0.0 (2021-02-23)
~~~~~~~~~~~~~~~~~~

* First release.