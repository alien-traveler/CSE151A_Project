# CSE 151A Group Project
## Preprocess steps
### Missing values
According to our data exploration, we found no missing values in our dataset, which means that missing values will not affect our model's performance. There is no need to drop missing values or use replacement data.

### Categorical data
We have some binary categorical data, such as ```GENDER```. So for those binary categorical data, we plan to use one-hot encoding before we train the model. There are also some categorical data without ordinal order, so we will think it's significant to decide to make it into one-hot encoding or choose the most frequent type and set other types as others to reduce the model complexity.

For the catigornial data like ```JOB```, we plan to find the most affecting type and use one-hot encoding and other will group to others so we will not largerly increase the model complaxity since there are more than 10 types of job in ```JOB```.

### Quantitive data
We have some quantitive data with large number and outliers. We will try to standerdize it before we can use in our model. We will also want to test out how outlier might affect our model's accuracy before we decide to drop outliers. We try to clean the data to make our mdoel have best preformance
