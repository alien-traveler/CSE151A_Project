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
- For NAME_EDUCATION_TYPE, we use ordinal encoding because the numbers have ordinal meaning. As the number gets larger, the person's education level gets higher. 
- For FLAG_PHONE, it is already binary, so we don't need to do anything.
- For FLAG_EMAIL, it is already binary, so we don't need to do anything.
- For NAME_FAMILY_STATUS, we use one hot encoding, since the family status is not ordinal
- For NAME_HOUSING_TYPE, we use one hot encoding, since the housing type is not ordinal
- For DAYS_BIRTH, we standardize the them because the data is -7489 and -24611, which is a large range, so we need to standardize so the model won't be influenced by its large magnitude. Additionally, according to our EDA, this is a normal distributed feature, so we use StandardScaler
- FOR DAYS_EMPLOYED, We used MinMaxScaler to scale the data. We did not use StandardScaler because the feature is highly skewed and we don't  the model won't be influenced by its large magnitude.
- For FLAG_MOBIL, it is already binary, so we don't need to do anything. I dropped it, since this feature is constant. In this case, including this feature does not make any sense, since it does not have any variability nor providing any predictive power.
- For FLAG_WORK_PHONE, it is already binary, so we don't need to do anything.
- For JOB, we use one hot encoding, since the job type is not ordinal.
- For BEGIN_MONTHS, we standardize the them because the data, ranging from 0 to -60, is in different scale to other numerical data, so we can better fit them in our model.
- For STATUS, we use ordinal encoding because the numbers have ordinal meaning. As the number gets larger, the person's status gets worse, indicating that the loan is more days overdue. We also use negative numbers to represent a better status, such as -1 to represent the loan is paid on time and -2 to represent no loan for that month.

## Model 1: Logistic Regression
By looking at our train error and test error, it seems like our model is performing extremely well by achieving 99% accuracy, 100% recall and precision [preprocess and model](preprocess_model.ipynb). Both the training error and the test error are extremely low with very little difference. This may be evidence that our model is within the ideal range of model complexity. However, it is very suspicious since our model is not very complicated and we've only applied basic feature engineering on the features. Therefore, a hypothesis that we have to justify the performance of our model is that the data lacks variability and the similarity in the dataset caused the training and testing datasets to be very similar, which leads to similar results between training and testing error. We try to look for any potential error in our preprocessing and splitting process, but they don't change the result that much. For example, we split the train/test data while ensuring the target variable retained its appropriate proportions (i.e., reflecting the significant imbalance with many false values and fewer true values). We also use k-fold cross validation to ensure the performance generalizes across different splits of the dataset. However, we are still unsure about why the performance is so well. Thus we will continue our investigation in the next milestone by testing with other models such as decision tree to see if the increased complexity in the model will change the performance on the dataset. We chose to test decision tree classifier next because it operates based on a very different logic than logistic regression and we can finetune the complexity of the model in fine detail to better carry out our experiment to find out whether the performance of the logistic regression came from model complexity or if there's an issue with the dataset itself. 

**Details on the model can be found in the `preprocess_model.ipynb` notebook.**

As mentioned above, the conclusion we reached at the end of training our first model is that it is performing extremely well. However, we are uncertain about the cause of such supreme performance. Based on our current result, there isn't really much room for improvement. Therefore, we are aiming to try out other different models to see if the high performance is due to a mistake in our pipeline. 

# Milestone4

As noted in our last milestone, even the simple logistic regression model with basic feature engineering yielded nearly identitical low train and test error. Therefore, in this milestone, the focus of our work is on investigating the cause of such model performance. Last time, we deployed stratification to make sure that our train dataset and test dataset both contained an equal proportion of 1s and 0s for `TARGET`, despite our effort, the first model still performed extremely well. This result ruled out the possibility that the good performance being coming from our test data not containing 1s (the near-perfect accuracy came from just taking advantage of the dataset's imbalance in `y` variable and predicting just the popular `TARGET `) .

In this milestone, the first thing we tried is to see how our first model would perform under different metrics, hoping that by changing the metric, we can unveal the cause of such high accuracy. We tried using precision, recall, and F1 score to measure the performance of our model, however, the result came back as all 1s, showing that our model's prediction on the test dataset is perfect. 

## Model 2: XGBoost classifier
The second model that we tried is XGBoost classifier. We chose this model because we wanted to utilize its `scale_pos_weight` parameter to adjust the weight for the minority class, ensuring the model accounts for class imbalance effectively. In addition, we employed GridSearchCV to fine-tune the model's hyperparameters, including n_estimators, learning_rate, max_depth, min_child_weight, and gamma. After optimizing our model, we once again found the training and testing accuracy of the model being 100%. If we look at this result alone, it means that we once again achieved perfect prediction of the problem, with no overfitting nor underfitting since our train and test accuracy are identical. Given the results, another model that we may try is RandomForest Classifier because its capability to handle high dimonsional data (what we have in this dataset), resistence to overfitting, and its capability to handle imbalanced data. 

However, we do not think the models should work like this without some problem going on. Very rarely do we get a perfect model that perfectly reveals the underlying patterns of the world, not to mention that we have two here. 

Next, we used `RandomOverSampler` to transform our imbalanced dataset into a balanced one and reapplied the balanced data to the model we developed in Milestone 3. This allowed us to evaluate how balancing the data could enhance the model's performance. The results showed that the balanced dataset yielded a confusion matrix with nearly equal True Positives (TP) and True Negatives (TN), which was a significant improvement compared to Milestone 3, where there was a high number of True Positives but very few True Negatives. Additionally, we also applied the balanced data with the new XGBClassifier model, which also demonstrated strong performance. 

**Details on the model can be found in the [test.ipynb](test.ipynb) notebook.**

In conclusion, our second model also demonstrated perfect performance by predicting not just train dataset, but also all items in the test dataset perfectly. Due to its perfect performance, we can't further improve it. Despite our effort to identify problem with how we delt with the dataset, the result remained the same. We did not see any column directly correlating to the target, we addressed the imbalance issue by using methods such as stratefication and `RandomOverSampler`. We are unable to find any other potential issue that may have cause such phenomenon from what we learned in this class and previous classes. Therefore, we believe the cause of this issue is because of the dataset itself being not diverse and representative of the real world data enough. Instead of revealing the underlying pattern of how credit card approval works in the real world, the dataset formed a certain pattern within itself that is not complicated. No matter how we split the data, the model still learns the underlying pattern of the dataset due to the simplicity of the underlying pattern within the dataset. 

# Milestone5

## Introduction
In this project, we tried to tackle the task of predicting credit card approvals using machine learning models. This project was chosen because of its practical significance in the financial industry and its relevance to real-world decision-making processes. Credit card approvals involve evaluating an applicant’s financial health and demographic information, and a good predictive model can help automate and streamline this process, reducing time and cost for institutions. What makes this project particularly exciting is the opportunity to work with a dataset that mirrors a real-world application, combining data preprocessing, feature engineering, and model evaluation to tackle a problem with immediate, tangible benefits. The challenge of handling class imbalance and diverse feature types adds complexity, making it a valuable learning experience and a chance to explore advanced techniques.

Having a robust predictive model for credit card approvals is important. It allows financial institutions to make consistent, data-driven decisions, reducing the likelihood of human errors or subjective biases. A reliable model can also improve accessibility by enabling quicker approvals for applicants who meet the criteria, fostering financial inclusion. Conversely, identifying potential risks accurately helps institutions maintain their financial stability, ultimately benefiting both the consumers and the financial ecosystem. In this project, we aim to investigate how various machine learning models perform in predicting credit card approvals, explore the impact of preprocessing techniques, and evaluate the limitations of the dataset. Through this work, we hope to highlight the critical role of representative data and thoughtful model design in achieving equitable and reliable outcomes in financial applications.
## Methods Section
### Preprocessing
#### Missing Values
We found no missing values in the dataset, eliminating the need for imputation or removal.

#### Categorical Data Encoding
For variables like `CODE_GENDER`, `FLAG_OWN_CAR`, and `FLAG_OWN_REALTY`, binary encoding was applied. Features with inherent ordinal relationships, such as `NAME_EDUCATION_TYPE`, were ordinally encoded (e.g., higher education levels received larger numerical values).Features like `NAME_FAMILY_STATUS`, `NAME_HOUSING_TYPE`, and `JOB` were one-hot encoded to capture non-ordinal relationships. For JOB, less frequent categories were grouped as "Other" to reduce model complexity.

#### Quantitative Data Transformation
Continuous variables like `AMT_INCOME_TOTAL`, `DAYS_BIRTH`, and `BEGIN_MONTHS` were standardized using either StandardScaler or MinMaxScaler, depending on their distributions. Also, we evaluated the effect of outliers on `AMT_INCOME_TOTAL` and other features, standardizing them to minimize their influence on the model while preserving all data points.

#### Feature Selection:
We removed the feature `FLAG_MOBIL` because it was constant across all rows, providing no predictive value.

#### Target Variable:
To address the class imbalance in the variable `Target`, we experimented with RandomOverSampler to create a balanced `Target` dataset.


### Method 1: Logistcal regression
For our first model, we utilized logistic regression due to its simplicity, interpretability, and effectiveness in binary classification problems such as predicting credit card approvals. Logistic regression models the probability of an applicant being approved or rejected based on the provided features by fitting a logistic function to the dataset.

We trained the logistic regression model on the preprocessed dataset after applying feature encoding, scaling, and handling class imbalance through stratified splitting.The target variable, TARGET, was used to classify applicants into two categories: approved or rejected. To ensure robust evaluation, we implemented k-fold cross-validation and monitored metrics such as accuracy, precision, recall, and F1-score to measure performance.

### Method 2: XGBoost Classifier
For the second model, we used the XGBoost Classifier, a powerful ensemble learning algorithm known for its ability to handle complex data patterns and imbalanced datasets effectively. XGBoost is particularly suitable for binary classification problems and allows fine-tuning of hyperparameters to achieve optimal performance.

#### Method 2.1: XGBoost Classifier with imbalanced dataset
When dealing with imbalanced dataset, we tried to utilizing the `scale_pos_weight` parameter to assign greater weight to the minority class, ensuring the model accounts for class imbalance during training. In addition, we employed GridSearchCV to fine-tune the model's hyperparameters, including n_estimators, learning_rate, max_depth, min_child_weight, and gamma. Lastly, cross-validation was used to ensure that the tuned hyperparameters generalized well across different data splits.

#### Method 2.2: XGBoost Classifier with balanced dataset
To address the class imbalance in the dataset, we applied RandomOverSampler to create a balanced dataset where the minority class (TARGET = 1) was oversampled to match the majority class (TARGET = 0). The balanced dataset was then used to train the XGBoost Classifier to evaluate how balancing the data impacts model performance and predictive power. Then, we followed the same steps as previous XGBosst training by running GridSearchCV to optimize key hyperparameters for the balanced dataset. This GirdSearchCV also includes n_estimators, learning_rate, max_depth, min_child_weight, and gamma. We also applied cross-validation to ensure the model’s generalization ability to unseen data

### Method 3: Investigating Dominant Features by Removing  `Status`
The motivation for this method stems from the fact that our previous models (logistic regression and XGBoost) achieved near-perfect performance, which raises questions about what drives such results. The detailed performance results will be analyzed in the next section. Through this method, we aim to investigate the underlying cause of this perfection by identifying and analyzing the most influential features in the dataset.

To achieve this, we calculated the correlation between each feature and the target variable (TARGET) to rank their importance. We observed that the feature Status had a relatively high correlation of 0.35 with TARGET, suggesting it might have a strong influence on the model's predictions. To test this hypothesis, we removed the feature Status and retrained both the logistic regression model and the XGBoost Classifier (with imbalanced and balanced datasets). For the XGBoost models, we re-optimized the hyperparameters using GridSearchCV to ensure the models performed optimally without the influence of the removed feature. By comparing the performance of these updated models to the original ones, we aim to determine whether the dominance of the Status feature was a contributing factor to the perfect performance observed earlier
## Results Section
**Incomplete: need figures**


For model1, the training metrics indicate exceptional performance. The model achieves 100% training accuracy, along with near-perfect precision, recall, and F1-scores across all classes. The confusion matrix further supports these results, showing only eight misclassifications in the majority class and none in the minority class. 
![image](imgs/cm_log_balance.png)


The second model also achieves perfect performance, with 100% accuracy, precision, recall, and F1-scores on both the training and test datasets. The confusion matrix confirms no misclassifications, demonstrating flawless generalization and error-free predictions across all classes. 
![image](imgs/cm_xgboost_balance.png)

After we removed the `Status` feature, the logistic regression model achieves an accuracy of 61.06%, with moderate precision, recall, and F1-scores for both classes. Class 0 has a slightly better recall (64%) compared to class 1 (58%), while class 1 has marginally better precision (62% vs. 60%). The confusion matrix highlights significant misclassifications in both classes, indicating room for improvement in the model's ability to generalize.

![image](imgs/cm_log_no_status.png)

Similarly, the XGBoost Classifier demonstrates a strong performance with a training error of 4.62% and a test error of 4.68%, indicating consistent generalization between training and testing. It achieves 95% accuracy, precision, recall, and F1-scores across both classes. The confusion matrix shows that most predictions are correct, with a small number of misclassifications: 6,833 for class 0 and 8,203 for class 1. Overall, the model performs well but has slight room for improvement in reducing errors.

![image](imgs/cm_no_status_xgb.png)

## Discussion Section
**Incomplete**

This may be evidence that our model is within the ideal range of model complexity. However, it is very suspicious since our model is not very complicated and we've only applied basic feature engineering on the features. 

Despite our effort to identify problem with how we delt with the dataset, the result remained the same.
## Conclusion

## Statement of Collaboration