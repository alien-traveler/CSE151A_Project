# CSE 151A Group Project 

# Milestone2
## Download the dataset
The dataset is available at [Kaggle](https://www.kaggle.com/datasets/laotse/credit-card-approval). You can download the dataset by clicking the download button on the right side of the page. For our project, we have saved the dataset as `credit_card_approval.csv`.

## Environment Requirement
We use the default environment of Google Colab, which is defined in `requirements.txt`

## Preprocess steps
### Missing values
According to our data exploration, we found no missing values in our dataset, which means that missing values will not affect our model's performance. There is no need to drop missing values or use replacement data.

### Categorical data
We have some binary categorical data, such as ```GENDER```. So for those binary categorical data, we plan to use one-hot encoding before we train the model. There are also some categorical data without ordinal order, so we can encode it with one-hot encoding or choose the most frequent type and set other types as others to reduce the model complexity.

For the catigornial data like ```NAME_EDUCATION_TYPE```, we plan to use ordinal encoding since there are some ordinal order in the data. For example, we might encode the data as 0 for lower secondary, 1 for secondary / secondary special, 2 for incomplete higher, 3 for higher education, and 4 for academic degree. 

For the catigornial data like ```JOB```, we plan to find the most affecting type and use one-hot encoding and other will group to others so we will not largerly increase the model complaxity since there are more than 10 types of job in ```JOB```.

### Quantitive data
We have explored the quantitive data in the dataset, and they are `AMT_INCOME_TOTAL`, `DAYS_BIRTH`, `DAYS_EMPLOYED`. The detailed information can be found at the bottom of `data_exploration.ipynb` notebook.
We have some quantitive data with large number and outliers, such as those in the "annual income" section. We will standerdize it before we can use in our model. We will also want to test out how outliers might affect our model's accuracy before we decide to drop outliers. We try to clean the data to make our mdoel have best preformance. 

**More details for each columns of the dataset can be found in the `data_exploration.ipynb` notebook.**

# Milestone3
## Preprocess Data
We have preprocessed columns: CODE_GENDER, FLAG_OWN_CAR, FLAG_OWN_REALTY, CNT_CHILDREN
AMT_INCOME_TOTAL, NAME_EDUCATION_TYPE, NAME_FAMILY_STATUS, NAME_HOUSING_TYPE, DAYS_BIRTH, DAYS_EMPLOYED, FLAG_MOBIL, FLAG_WORK_PHONE, FLAG_PHONE, FLAG_EMAIL, JOB, BEGIN_MONTHS, STATUS. 

- For CODE_GENDER, we encoded 0 for female and 1 for male.
- For FLAG_OWN_CAR, we encoded 0 for no and 1 for yes.
- For FLAG_OWN_REALTY, we encoded 0 for no and 1 for yes.
- For CNT_CHILDREN, we encoded 0 for "No children", 1 for "1 children", 2 for "2+ children".
- For AMT_INCOME_TOTAL, we standardize the data because the income (unit in thousands) is in different scale to other numerical data, so we can better fit them in our model.
- For NAME_EDUCATION_TYPE, we use one-hot encoding, and we have Education_Academic degree,Education_Higher education, Education_Incomplete higher, Education_Lower secondary, Education_Secondary / secondary special.

- For FLAG_PHONE, it is already binary, so we don't need to do anything.
- For FLAG_EMAIL, it is already binary, so we don't need to do anything.
- For JOB, **TBD**
- For BEGIN_MONTHS, we standardize the them because the data, ranging from 0 to -60, is in different scale to other numerical data, so we can better fit them in our model.
- For STATUS, we use ordinal encoding because the numbers have ordinal meaning. As the number gets larger, the person's status gets worse, indicating that the loan is more days overdue. We also use negative numbers to represent a better status, such as -1 to represent the loan is paid on time and -2 to represent no loan for that month.