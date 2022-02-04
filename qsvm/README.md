# Basic QSVM model

## TODO 
* TODO: Have the mean ROC (solid) + the individual ROCs per fold (thiner line) + 1 std band (transparent).
* TODO: Implement everything for Kinga's results: data loading from a file and not qdata+AE, test statistic plot, classical SVM benchmark, also in the repo (currently in local branch + jupytor nb).
Notes:
- ntrain/ntest/nvalid arguments refer to the total  signal+background samples. The data sets are always balanced, i.e., signal (50%) + background (50%).
- `test.py` is used to test a trained (Q)SVM model. To do that with the current implementation of `kernel='precomputed'`, one needs to compute a (Gram) matrix K_test = K(X_test, X_traist) that is of expected shape (n_test, n_train) (`scikit-learn` requirement, [docs](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC.predict)). This means, that in order to not get unexpected behaviour the X_train used during training needs to be the same with the one used in testing. This can be ensured by using the same `seed` to `qdata`..
- To save sklearn models joblib package is used. Serialization and de-serialization of objects is python-version sensitive. 

Alternatives: As of Python 3.8 and numpy 1.16, pickle protocol 5 introduced in PEP 574 supports efficient serialization and de-serialization for large data buffers natively using the standard library:

pickle.dump(large_object, fileobj, protocol=5)
    
