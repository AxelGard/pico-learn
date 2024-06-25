# pico-learn
A super small version of sklearn implemented using numpy


<img src="https://raw.githubusercontent.com/AxelGard/pico-learn/master/doc/pico.jpeg" alt="drawing" style="width:300px;"/>

This is not **really** made for you to use, mainly for learning purposes.

The inspiration for this project came from [tinygrad](https://github.com/tinygrad/tinygrad) and [micrograd](https://github.com/karpathy/micrograd), but instead of focusing on neural networks, Pico focuses on more classic machine learning, similar to [scikit-learn](https://github.com/scikit-learn/scikit-learn).

The project is named pico due to how small and incomplite it is. 

## Usage 

```python
import numpy as np
from picolearn.linear import LinearRegrasion 


X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
y = np.dot(X, np.array([1, 2])) + 3
X_test = np.array([[3, 5]])


model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X_test)

```

## Model implemented

this are the model that have been implemented and is planed to be implemented

- [x] Linear Regreassion
- [x] KNN Regreassion
- [x] KNN Classifier 
- [ ] Support Vector Machine Classifier
- [ ] Decision Tree Classifier


## Implementation validation 

Each model is implemented then compared to Sklearns equivelent model. 
This can be seen in the [tests](https://github.com/AxelGard/pico-learn/tree/master/test).

## install 

Due to pico is made for learning purposes I have not added it to PyPi.

But if you want to try it out this is how: 

clone the repo 
```bash
git clone https://github.com/AxelGard/pico-learn.git && cd pico-learn
```

setup a python env
```bash
python3 -m venv env && source env/bin/activate
```

install dependacies and pico 
```bash
pip install -r ./requirements.txt && pip install -e .
```

