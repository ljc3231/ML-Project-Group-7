# ML-Project-Group-7
## Group Members
- Liam Cummings
- Sam Cordry
- Dominic Vinciulla

## Abstract
This is the repository for our CSCI-335 machine learning course. Our models aim to classify
network traffic as anomalous or normal. To do this, we have employed a number of techniques.
### Preprocessing
- Z-Score normalization
- One-Hot encoding for categorical features
- Mean-replaced missing values
### Modeling
- Support Vector Machine
- Isolation Forest
- K-Nearest Neighbors
### Evaluation
- K-Fold Cross Validation

## How to Run
Our models can be ran through our driver.py file:
\> python3 ./code/driver.py full
-> will run the 3 models with the full kdd-cup-99 dataset

\> python3 ./code/driver.py partial
-> will run the 3 models with kdd-cup-99-10percent dataset
