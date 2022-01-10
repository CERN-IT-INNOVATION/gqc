# Basic QSVM model

## TODO 
* TODO: Include hyperparameter optimisation for C (maybe splitting of is needed).
* TODO: Have the mean ROC (solid) + the individual ROCs per fold (thiner line) + 1 std band (transparent).

Sidenote:
    To save sklearn models joblib package is used. Serialization and
    de-serialization of objects is python-version sensitive.

    Alternatives: As of Python 3.8 and numpy 1.16, pickle protocol 5
    introduced in PEP 574 supports efficient serialization and de-serialization
    for large data buffers natively using the standard library:
         pickle.dump(large_object, fileobj, protocol=5)
    
