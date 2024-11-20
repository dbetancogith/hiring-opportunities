# Hiring Opportunities for Minorities 
Dario Betancourt

#### Executive summary

Increasing access to job opportunity for minorities and women is crucial for reducing well- documented race, ethnicity, and gender gaps in the economy. While a proliferation of initiatives related to diversity, equity, and inclusion speak to firms’ interest in these issues, a persistent doubt remains: how can firms increase diversity without sacrificing quality? (Danielle Li, et al June 1, 2014).

The main findings of this project demonstrate that expanding diversity in the workplace does not necessarily involve a tradeoff between equity and efficiency. In particular, current recruiting practices seem to fall short of their full potential, leaving significant room for new ML tools to simultaneously enhance hiring rates and demographic representation.

### Content Index 
- [Rationale](#rationale)
- [Research Question](#research-question)
- [Data](#data)
- [Methodology](#methodology)
- [Baseline Model](#baseline-model)
- [Comparing Models](#comparing-models)
- [Permutation Importance](#permutation-importance)
- [Adjusting the performance metrics](#adjusting-the-performance-metrics)
- [Next Steps and Recommendations](#next-steps-and-recommendations)
- [Outline of project](#outline-of-project)
- [Contact and Further Information](#contact-and-further-information)

#### Rationale

Increasing access to job opportunities for minorities and women is crucial for reducing well- documented race, ethnicity, and gender gaps in the economy. 		

**Bias Towards the Majority Class:** Machine learning models, especially those that optimize for accuracy, may become biased toward predicting the majority class. A model can achieve a high accuracy simply by predicting the majority class for all samples, even though it fails to capture the minority class properly.
Example: In a dataset with 95% negative class and 5% positive class, a model that predicts "negative" for every instance would have 95% accuracy but would fail to identify any of the positive cases, which might be the most critical part of the problem (e.g., detecting fraud, rare diseases).

#### Research Question

How to reduce the bias against Black and Hispanic in the hiring process without sacrificing quality? 

Maintaing quality by considering the most optimistic assessment of their hiring potential. That is, among applicants with **the same predicted hiring potential** the exploration will give opportunity to applicants in the minorities.

### Data:
The dataset is sourced from Kaggle.
Link: https://www.kaggle.com/datasets/rabieelkharoua/predicting-hiring-decisions-in-recruitment-data

This dataset provides insights into factors influencing hiring decisions. Each record represents a candidate with various attributes considered during the hiring process including ethnicity.

The imbalance in ethnicity within the dataset introduces a bias, disproportionately affecting minority groups' chances of being hired. This underrepresentation may lead the model to favor the majority group, ultimately reducing fairness in hiring predictions.

#### Methodology

1. Set the baseline with Logistic Regression (LASSO)
2. Analyze three resampling methods for balancing datasets
   - Oversampling the minority class using SMOTE
   - Undersampling the majority class with RandomUnderSampler
   - Adjusting Class Weights as a parameter for the model
3. Each method is applied to four selecting models
    - LogisticRegression 
    - SVC 
    - DecisionTreeClassifier
    - RandomForest 
    therefore there are 12 models 
4. Get the metrics accuracy, precision, recall, and ROC for train and test data
5. Compare the test predictions with original data and the baseline - plot the results in a pie
6. Adjust the metrics with comparing Cumulative Gain Curves (LIFT charts) and ROC
7. Review the permutation importance for each model, to demonstrate the balancing methods do not affect the features importance

### Baseline Model:

The base model is Logistic Regression with default parameters, it has an accuracy of 0.81.
Compared with the original data the number of Black and Hispanic people went from 5.0% to 5.8% and 7.0%% to 7.1% showing an increase of 0.8% and 0.1 respectively for each ethnicity.

The second model combines Logistic Regression with LASSO regularization has an accuracy 0.81, and an increase in the Back and Hispanic participation of 0.8% and 0.7% as shown in the graph below. Also train data shows how the model can reduce the participation of the Hispanic people by 0.6%.

![LASSO Baseline](/images/1LASSOBaseline.png)

### Comparing Models 

1. Running the models with SMOTE - Synthetic Minority Over-sampling Technique - the results are:

   - Random forest has the highest ROC value of 0.89 followed by SVC with 0.87
   - Random forest accuracy 0.90, SVC accuracy 0.85
   - The fastest model was Logistic Regression with 0.008

    The graphic below shows metrics comparison for the models with SMOTE 
   
    ![metric-smote](/images/metric-smote.png)

    According to these results, SMOTE did not sacrifice the performance of the model and the Black and Hispanic hired people increased from 5.0% to 6.2% and 7.0% to 9.3% on the SVC respectively. **The increase was even better on the Decision Tree Classifier model Black hired people increase on 2.5% to 7.5% and Hispanic on 4.3% to 11.3%**

    ![pie-smote](/images/pie-smote.png)

2. Under-Sampling - RandomUnderSampler. These are the results for this technique:

    - Random forest has the highest ROC value of 0.88 followed by Logistic Regression with 0.86
    - Random forest accuracy 0.86, Logistic Regression accuracy 0.80
    - The fastest model was Decision Tree with 0.003

    The graphic below shows metrics comparison for the models with Under-Sampling 

    ![metric-under-sampling](/images/metric-under-sampling.png)

    **Under-Sampling results showed great performance in all models, and Black and Hispanic hired people increased from 5.0% to 6.5% and 7.0% to 10.1% respectively with Random Forest.** The Decision Tree Classifier showed an increase of 2.7% for Black candidates to 7.9% and 2.3% for Hispanic to 9.3%

    ![pie-under-sampling](/images/pie-under-sampling.png)

3. Balancing Weights all 4 algorithms used accept the parameter class_weight set to ‘balanced’ here are the results:

   - Random forest has the highest ROC value of 0.89 followed by SVC with 0.86
   - Random forest accuracy 0.89, SVC accuracy 0.86
   - The fastest model was Decision Tree Classification with 0.003

    The graphic below shows metrics comparison for the models with Balancing Weights 

    ![metric-balancing](/images/metric-balancing.png)

    Logistic Regression showed the biggest increase fo Black and Hispanic candidates with 6.4% and 9.0%, these results showed that for increasing the participation of Black and Hispanic candidates and in general underrepresented classes it is better to combine this method with a resampling technique. 

    ![pie-balancing](/images/pie-balancing.png)
    
### Permutation Importance
This machine learning technique is used to assess the importance of individual features within a model by randomly shuffling the values of a single feature and observing how much the model's performance degrades. 

The graphs below show that the resampling or balancing of the data did not affect the importance of the features (SMOTE). 

![7permutation](/images/7permutation.png)

### Adjusting the performance metrics

The Cumulative Gain Curve shows how well the model can rank instances in order of their likelihood to respond positively (e.g., get hired).

On the other hand The ROC Curve (Receiver Operating Characteristic curve) is a useful metric for marketing campaigns in classification models because it helps evaluate the model's ability to distinguish between classes (e.g., potential hires vs no-hires) across different decision thresholds. Here’s why the ROC is valuable in this context.

The ROC curve plots the True Positive Rate (TPR) (also called sensitivity or recall) against the False Positive Rate (FPR) at various classification thresholds.

Next, The Lift Curve is derived from the cumulative gains chart; the values on the y axis correspond to the ratio of the cumulative gain for each curve to the baseline. Thus, the lift at 10% for the category Yes is 50%/10% = ~5.0 for most models. It provides another way of looking at the information in the cumulative gains chart.

SMOTE
![8roc-curves](/images/8roc-curves.png)

Plotting The Calibration Curves of a classifier is useful for determining whether or not you can interpret their predicted probabilities directly as a confidence level. For instance, a well-calibrated binary classifier should classify the samples such that for samples to which it gave a score of 0.8, around 80% should actually be from the positive class.

![9calibration](/images/9calibration.png)
    
Finally The Confusion Matrix for each model shows the values for TP, FP and FN, in this case the idea is to increase the number of people in the campaign that will say Yes and reduce the number of calls.

![10conf-smote](/images/10conf-smote.png)

![11conf-under](/images/11conf-under.png)

![12conf-balancing](/images/12conf-balancing.png)

### Next Steps and Recommendations
 
- There are balancing methods that could be tested such as Adaptive Synthetic Sampling (ADASYN) 
- Include other minorities such as gender and age
- One of the algorithms to try should be the Upper Confidence Bound - Bandit
- Since most ATS systems have the ML models at the beginning of the funnel, it would be interesting to run these models with a dataset of selection for interview, and track the people hired at end. 
- Tunning the algorithms and research on some of the behaiviors such the the one for decision tress classification in the calibrarion graphic for SMOTE models

#### Outline of project

- Link to notebook [Hiring Opportunities](https://github.com/dbetancogith/hiring-opportunities/blob/main/capV12.ipynb)
- Dataset [Dataset](https://www.kaggle.com/datasets/rabieelkharoua/predicting-hiring-decisions-in-recruitment-data)
- Download the dataset [Download](https://github.com/dbetancogith/hiring-opportunities/blob/main/data/Job_Applicants_by_Ethnicity.csv)

##### Contact and Further Information

Email: dariobz3071@gmail.com

