# Following showcase for QDisCoCirc

Build an app to test out the models trained in the QDisCoCirc experimental paper.
To run the app first install the project
```
pip install .
```
Then run the server:
```
fastapi dev app.py
```
You can then access the page on `localhost:8000`.


Note that this is intended as a lightweight (in terms of dependencies) implementation,
and thus will be less efficient at evaluating circuits as no optimisations have been included.
