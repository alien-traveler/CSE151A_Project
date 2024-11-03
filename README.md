# CSE 151A Group Project
## Preprocess steps
### Missing values
According our data exploration, we found no missing values in our dataset, which means that missing value will not affecting our model's preformance. It is no need to drop missing values or use any replacement data.

### Catigorical data
We have some catigorical data that is binery, such as ```GENDER```. So for those binery catigorical data, we plan to use one-hot encoding before we train the model. And there are also some catigorical data that without ordinal order, so we will think about it's significants to decide to make it into one-hot encoding or choose the most frequance type and set other types as other to reduce the model complaxity.

For the catigornial data like ```JOB```, we plan to find the most affecting type and use one-hot encoding and other will group to others so we will not largerly increase the model complaxity since there are more than 10 types of job in ```JOB```.

### quantitive data
We have some quantitive data with large number and outliers. We will try to standerdize it before we can use in our model. We will also want to test out how outlier might affect our model's accuracy before we decide to drop outliers. We try to clean the data to make our mdoel have best preformance.
