### Dependancies:
- numpy
- scipy
- pandas
- scikit-learn
- xgboost==0.4a30

### Data creation (Put the original data in the same folder)
```python3 create_data.py data.pkl```

### Run different model on the data
```python3 train_lr.py data.pkl sub_lr.csv```
```python3 train_rf.py data.pkl sub_rf.csv```
```python3 train_xgb.py data.pkl sub_xgb.csv```
```python3 train_dnn.py data.pkl sub_dnn.csv```


### TODOs:
1. Complex feature engineering
2. Model stacking

