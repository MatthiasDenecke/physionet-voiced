# Final Report

## 1. Problem Statement


## 2. Exploratory Data Analysis

### Dataset

The dataset contains 13 features, one predicted variable
and 110527 records.

### Features

The features in the dataset are

| Feature Name  | Explanation| Values
|:--------------|:-----------|:------
|PatientId      | unique patient ID | Strings
|AppointmentID  | unique appointment ID | Strings
|Gender         |  patient's gender | F,M
|ScheduledDay   |  when the appointment was scheduled | Date and Time 
|AppointmentDay | date of the appointment | Date
|Age            | patient's age | Integer between 0 and 115
| Neighbourhood | patient's neighbourhood (in Brazil) | The name of one of 81 neighbourhoods
| Scholarship   | whether the patient benefited from social welfare to pay for the appointment|0,1
| Hipertension  | whether the patient has hypertension|0,1
| Diabetes      |  whether the patient has diabetes|0,1
| Alcoholism    | whether the patient is an alcoholic|0,1
| Handcap       | whether the patient is handicapped|0,1,2,3,4
| SMS_received  | whether the patient received a reminder by text message|0,1

Here is a glimpse of the dataset:

| | Gender   |  ScheduledDay | AppointmentDay | Age   | Neighbourhood | Scholarship | Hipertension | Diabetes | Alcoholism | Handcap | SMS_received No-show
|-|:---------|:--------------|:---------------|:------|:--------------|:------------|:-------------|:---------|:-----------|:--------|:------------|:-----------
|0|  F     | 2016-04-29T18:38:08Z | 2016-04-29T00:00:00Z | 62  | JARDIM DA PENHA   |  0 |           1    |         0     |    0  |         0  |      0      |       No
|1|  M     | 2016-04-29T16:08:27Z | 2016-04-29T00:00:00Z | 56  | JARDIM DA PENHA   |  0 |           0    |         0     |    0  |         0  |      0      |       No
|2|  F     | 2016-04-29T16:19:04Z | 2016-04-29T00:00:00Z | 62  | MATA DA PRAIA     |  0 |           0    |         0     |    0  |         0  |      0      |       No
|3|  F     | 2016-04-29T17:29:31Z | 2016-04-29T00:00:00Z |  8  | PONTAL DE CAMBURI |  0 |           0    |         0     |    0  |         0  |      0      |       No
|4|  F     | 2016-04-29T16:07:23Z | 2016-04-29T00:00:00Z | 56  | JARDIM DA PENHA   |  0 |           1    |         1     |    0  |         0  |      0      |       No

Of these, the patient ID and the appointment ID do not
contain any information relevant for classification and
are therefore dropped.

Following is an overview of the distribution of the different variables:

|    |           Age |   Scholarship   |  Hipertension  |  Diabetes     | Alcoholism     |     Handcap |  SMS_received  |    DayOfWeek  |     Day        |     Month    |      Delay     |
|:---|:--------------|:----------------|:---------------|:--------------|:---------------|:------------|:---------------|:--------------|:---------------|:-------------|:---------------|
count|  110527.000000|  110527.000000  | 110527.000000  |110527.000000  |110527.000000   |110527.000000|110527.000000   | 110527.000000 | 110527.000000  |110527.000000 | 110527.000000  |
mean |      37.088874|       0.098266  |     0.197246   |    0.071865   |    0.030400    |   0.022248  |      0.321026  |     1.851955  |     14.252228  |     4.823871 |     10.183702  |
std  |      23.110205|       0.297675  |     0.397921   |    0.258265   |    0.171686    |   0.161543  |      0.466873  |     1.378520  |      9.345629  |     0.715801 |     15.254996  |
min  |      -1.000000|       0.000000  |     0.000000   |    0.000000   |    0.000000    |   0.000000  |      0.000000  |     0.000000  |      1.000000  |     1.000000 |     -6.000000  |
25%  |      18.000000|       0.000000  |     0.000000   |    0.000000   |    0.000000    |  0.000000   |     0.000000   |    1.000000   |     6.000000   |    4.000000  |     0.000000   |
50%  |      37.000000|       0.000000  |     0.000000   |    0.000000   |    0.000000    |   0.000000  |      0.000000  |     2.000000  |     13.000000  |     5.000000 |      4.000000  |
75%  |      55.000000|       0.000000  |     0.000000   |    0.000000   |    0.000000    |   0.000000  |      1.000000  |     3.000000  |     22.000000  |     5.000000 |     15.000000  |
max  |     115.000000|       1.000000  |     1.000000   |    1.000000   |    1.000000    |   4.000000  |      1.000000  |     5.000000  |     31.000000  |    12.000000 |    179.000000  |



### Univariate Feature Distributions

#### Binary Variables

|    |    |    | 
|:--:|:--:|:--:| 
| ![eda/histogram-Alcoholism.png][hist_3]    | ![eda/histogram-Diabetes.png][hist_7]     | ![eda/histogram-Gender.png][hist_8]        | 
| ![eda/histogram-Hipertension.png][hist_10] | ![eda/histogram-Scholarship.png][hist_14] | ![eda/histogram-SMS_received.png][hist_15] |

It can be seen that the binary variables are all imbalanced.

#### Categorical Variables

![eda/histogram-Handcap.png][hist_9]

Like the binary variables, the handicap variable is imbalanced as well.

### Numerical Variables

![eda/histogram-Age.png][hist_2]
![eda/histogram-Neighbourhood.png][hist_12]



[hist_2]: eda/histogram-Age.png ""
[hist_3]: eda/histogram-Alcoholism.png ""
[hist_7]: eda/histogram-Diabetes.png ""
[hist_8]: eda/histogram-Gender.png ""
[hist_9]: eda/histogram-Handcap.png ""
[hist_10]: eda/histogram-Hipertension.png ""
[hist_12]: eda/histogram-Neighbourhood.png ""
[hist_14]: eda/histogram-Scholarship.png ""
[hist_15]: eda/histogram-SMS_received.png ""

The distribution of age is mostly even between 5 and 65 years which is to be expected. 
The number of appointments reduces over age 65 which may correlate with the fact that
patients start to die around that age. Likewise, the spike of appointments at age 0 may indicate
that newborns require more and more intensive care.

The distribution of the neighborhoods follows Zip's law.


### Predicted Variable

The predicted variable is No-Show indicating whether
the patient appeared to the appointment, and can take
values 'yes' or 'no'.

The distribution of the predicted variable is skewed as
shown in the table and figure below, indicating that
about one fifth of the patients miss their appointment.

Value | Counts | Relative
:-----|:-------|--------
No    | 88208  |79.81 %
Yes   | 22319  |20.19 %

![histogram_no_show][hist_13]

[hist_13]: eda/histogram-No-show.png ""

The strong imabalance in the dataset requires extra precaution during 
training to be taken.

## 3. Data Cleaning

Several records in the dataset contain illegal or otherwise
invalid values. One row contains a negative age. Abput 3500
rows contain an age of 0 which is considered valid.

Furthermore, 5 rows contain appointment dates which are earlier
than the day on which they were scheduled. These were removed
as well.

## 4. Feature Engineering

### Added Features

A new feature called 'delay' was added indicating the number of 
days between the day the appointment was scheduled and the actual
appointment. Further, the day, month and day of week of the appointment
day were added to the features.


### Univariate Feature Distributions of the Added Features

#### Categorical Variables

![eda/histogram-DayOfWeek.png][hist_5]
![eda/histogram-Month.png][hist_11]

The added features are distributed more evenly than the original binary features
present in the data. 

### Numerical Variables


![eda/histogram-Day.png][hist_4]

The day of the month is distributed unevenly. There are reduced appointments
around days 7 and 8, 14 and 15 and 21 through 23, indicating that these
days could be weekends during which less doctor's appointments are scheduled.

![eda/histogram-Delay.png][hist_6]

The distribution of the delay variable has an extremely long tail. For this reason,
delays greater than 60 days will be set to 60.




[hist_2]: eda/histogram-Age.png ""
[hist_3]: eda/histogram-Alcoholism.png ""
[hist_4]: eda/histogram-Day.png ""
[hist_5]: eda/histogram-DayOfWeek.png ""
[hist_6]: eda/histogram-Delay.png ""
[hist_7]: eda/histogram-Diabetes.png ""
[hist_8]: eda/histogram-Gender.png ""
[hist_9]: eda/histogram-Handcap.png ""
[hist_10]: eda/histogram-Hipertension.png ""
[hist_11]: eda/histogram-Month.png ""
[hist_12]: eda/histogram-Neighbourhood.png ""
[hist_14]: eda/histogram-Scholarship.png ""
[hist_15]: eda/histogram-SMS_received.png ""

### Correlation
The correlation between features is shown here.

![correlation][c1]

[c1]: eda/correlation.png "correlation"

It can be seen that most features are uncorrelated or weakly correlated which seems
to indicate that most features may potentially contribute to the classification power. 
Therefore, and together with the fact that there are not may features in the dataset 
to the number of examples, no dimensionality reduction was attempted.

### Feature Importance

A decision tree learns how to best split the dataset into smaller 
subsets before finally determining the prediction. 
Based on the impurity criterion, a tree can be built by greedily picking the 
features that contribute to the most information gain for eaach split. 
It is then possible to determine the importance of the features by looking 
 at rge order in which they were picked up. 


Shown below is the feature importance for a random forest classifier with 
impurity criterion set to GINI.


![Feature Importance][ft_1]

[ft_1]: eda/feature_importance.png ""

It can be seen that the four added features (described above) are in the 
top 7 important features, confirming that reasoning for adding them.

### Feature Encoding

The following features were used without further processing:

`'Scholarship','Hipertension','Diabetes','Alcoholism','SMS_received'`

The following features were one-hot encoded:

`'Neighbourhood','Handicap'`

The following features were scaled to a floating point number between
0 and 1:

`'Age','Delay','Day','DayOfWeek','Month'`


## Training Results

### Comparison of Classifiers on Cross-Validation

In the following, various scores for the different classifiers, trained with 10 fold cross-validation on the training
set, with and without resampling, are shown.

![accuracy text][p1]
![precision text][p2]
![recall text][p3]
![roc_auc text][p4]
![f1 text][p5]
![time text][p6]

[p1]: comparisons/accuracy.png "Accuracy"
[p2]: comparisons/precision.png "Precision"
[p3]: comparisons/recall.png "Recall"
[p4]: comparisons/roc_auc.png "ROC Area Under Curve"
[p5]: comparisons/f1.png "F1"
[p6]: comparisons/time.png "Time"


### Comparison of Classifiers on Train/Test Set

In the following, various scores for the different classifiers, trained on the entire
training set and evaluated against the test set, are shown.

![accuracy text][pt1]
![precision text][pt2]
![recall text][pt3]
![roc_auc text][pt4]
![f1 text][pt5]
![time text][pt6]

[pt1]: comparisons/accuracy.test.png "Accuracy"
[pt2]: comparisons/precision.test.png "Precision"
[pt3]: comparisons/recall.test.png "Recall"
[pt4]: comparisons/roc_auc.test.png "ROC Area Under Curve"
[pt5]: comparisons/f1.test.png "F1"
[pt6]: comparisons/time.test.png "Time"

### ROC-AUC and Confusion Matrices

Logistic regression resampled with random oversampling and the multilayer perceptron
resampled with balanced batch are among the best performers. Here are their
ROC-AUC graphs and confusion matrices:

#### Logistic Regression Learning Curve

The following shows the learning curve for logistic regression on the train dataset
with 10 fold cross validation and random oversampling.


![lr learning curve text][lc1]

[lc1]: results/lr_learning_cur.png "Logistic regression learning curve"


#### Logistic Regression Confusion Matrix

![lr conf mat text][r1]

[r1]: results/lr-confm.png "Logistic regression conf matrix"


#### Logistic Regression ROC-AUC Curve

![lr roc text][r2]

[r2]: results/lr-roc-curve.png "Logistic regression roc-auc"


#### Logistic Regression CAP-AUC Curve

![lr cap text][cap2]

[cap2]: results/lr-cap-curve.png "lr roc-auc"

#### Multilayer Perceptron Accuracy and Loss

The following shows the accuracy and loss for the training of the multilayer perceptron 
with 10 fold cross validation on the train dataset and balanced batch resampling.


![lr learning curve text][lm1]

[lm1]: results/loss-and-accuracy.cv.png "MLP accuracy and loss"


The following shows the accuracy and loss for the training of the multilayer perceptron 
on the entire train dataset and balanced batch resampling evaluated against
the test set.


![lr learning curve text][lm2]

[lm2]: results/loss-and-accuracy.test.png "MLP accuracy and loss"


#### MLP Confusion Matrix

![mlp conf mat text][r3]

[r3]: results/mlp-confm.png "mlp conf matrix"


#### MLP ROC-AUC Curve

![mlp roc text][r4]

[r4]: results/mlp-roc-curve.png "mlp roc-auc"


#### MLP CAP-AUC Curve

![mlp cap text][cap4]

[cap4]: results/mlp-cap-curve.png "mlp roc-auc"

