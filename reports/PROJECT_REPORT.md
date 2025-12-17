# 📊 Wellness Tourism Prediction: Insights

## 1. Executive Summary
This report captures the key data characteristics and the drivers of customer conversion.

## 2. Exploratory Data Analysis (EDA)

### Target Distribution
![Target Distribution](target_dist.png)

**💡 Observation:** The dataset is imbalanced. Only 19.3% of customers purchase the package. This confirms the need for F1-Score optimization over simple Accuracy.


### Correlation Matrix
![Correlation Matrix](correlation.png)

**💡 Observation:** We observe strong correlations between 'MonthlyIncome' and 'Age'. There is also a notable relationship between 'Passport' possession and 'ProdTaken'.


### Univariate Distribution of Age
![Univariate Distribution of Age](univariate_Age.png)

**💡 Observation:** The distribution of **Age** shows its central tendency and spread. Box plot indicates potential outliers.


### Univariate Distribution of DurationOfPitch
![Univariate Distribution of DurationOfPitch](univariate_DurationOfPitch.png)

**💡 Observation:** The distribution of **DurationOfPitch** shows its central tendency and spread. Box plot indicates potential outliers.


### Univariate Distribution of NumberOfPersonVisiting
![Univariate Distribution of NumberOfPersonVisiting](univariate_NumberOfPersonVisiting.png)

**💡 Observation:** The distribution of **NumberOfPersonVisiting** shows its central tendency and spread. Box plot indicates potential outliers.


### Univariate Distribution of PreferredPropertyStar
![Univariate Distribution of PreferredPropertyStar](univariate_PreferredPropertyStar.png)

**💡 Observation:** The distribution of **PreferredPropertyStar** shows its central tendency and spread. Box plot indicates potential outliers.


### Univariate Distribution of NumberOfTrips
![Univariate Distribution of NumberOfTrips](univariate_NumberOfTrips.png)

**💡 Observation:** The distribution of **NumberOfTrips** shows its central tendency and spread. Box plot indicates potential outliers.


### Univariate Distribution of NumberOfChildrenVisiting
![Univariate Distribution of NumberOfChildrenVisiting](univariate_NumberOfChildrenVisiting.png)

**💡 Observation:** The distribution of **NumberOfChildrenVisiting** shows its central tendency and spread. Box plot indicates potential outliers.


### Univariate Distribution of MonthlyIncome
![Univariate Distribution of MonthlyIncome](univariate_MonthlyIncome.png)

**💡 Observation:** The distribution of **MonthlyIncome** shows its central tendency and spread. Box plot indicates potential outliers.


### Univariate Distribution of PitchSatisfactionScore
![Univariate Distribution of PitchSatisfactionScore](univariate_PitchSatisfactionScore.png)

**💡 Observation:** The distribution of **PitchSatisfactionScore** shows its central tendency and spread. Box plot indicates potential outliers.


### Univariate Count of TypeofContact
![Univariate Count of TypeofContact](countplot_TypeofContact.png)

**💡 Observation:** The count plot for **TypeofContact** illustrates the frequency of each category within the dataset. Some categories are dominant.


### Univariate Count of CityTier
![Univariate Count of CityTier](countplot_CityTier.png)

**💡 Observation:** The count plot for **CityTier** illustrates the frequency of each category within the dataset. Some categories are dominant.


### Univariate Count of Occupation
![Univariate Count of Occupation](countplot_Occupation.png)

**💡 Observation:** The count plot for **Occupation** illustrates the frequency of each category within the dataset. Some categories are dominant.


### Univariate Count of Gender
![Univariate Count of Gender](countplot_Gender.png)

**💡 Observation:** The count plot for **Gender** illustrates the frequency of each category within the dataset. Some categories are dominant.


### Univariate Count of ProductPitched
![Univariate Count of ProductPitched](countplot_ProductPitched.png)

**💡 Observation:** The count plot for **ProductPitched** illustrates the frequency of each category within the dataset. Some categories are dominant.


### Univariate Count of MaritalStatus
![Univariate Count of MaritalStatus](countplot_MaritalStatus.png)

**💡 Observation:** The count plot for **MaritalStatus** illustrates the frequency of each category within the dataset. Some categories are dominant.


### Univariate Count of Passport
![Univariate Count of Passport](countplot_Passport.png)

**💡 Observation:** The count plot for **Passport** illustrates the frequency of each category within the dataset. Some categories are dominant.


### Univariate Count of OwnCar
![Univariate Count of OwnCar](countplot_OwnCar.png)

**💡 Observation:** The count plot for **OwnCar** illustrates the frequency of each category within the dataset. Some categories are dominant.


### Univariate Count of Designation
![Univariate Count of Designation](countplot_Designation.png)

**💡 Observation:** The count plot for **Designation** illustrates the frequency of each category within the dataset. Some categories are dominant.


### Multivariate Age vs. ProdTaken
![Multivariate Age vs. ProdTaken](box_Age_prod_taken.png)

**💡 Observation:** A comparison of **Age** distribution between buyers and non-buyers reveals potential differentiating factors. Higher Age seems to correlate with ProdTaken.


### Multivariate DurationOfPitch vs. ProdTaken
![Multivariate DurationOfPitch vs. ProdTaken](box_DurationOfPitch_prod_taken.png)

**💡 Observation:** A comparison of **DurationOfPitch** distribution between buyers and non-buyers reveals potential differentiating factors. Higher DurationOfPitch seems to correlate with ProdTaken.


### Multivariate NumberOfPersonVisiting vs. ProdTaken
![Multivariate NumberOfPersonVisiting vs. ProdTaken](box_NumberOfPersonVisiting_prod_taken.png)

**💡 Observation:** A comparison of **NumberOfPersonVisiting** distribution between buyers and non-buyers reveals potential differentiating factors. Higher NumberOfPersonVisiting seems to correlate with ProdTaken.


### Multivariate PreferredPropertyStar vs. ProdTaken
![Multivariate PreferredPropertyStar vs. ProdTaken](box_PreferredPropertyStar_prod_taken.png)

**💡 Observation:** A comparison of **PreferredPropertyStar** distribution between buyers and non-buyers reveals potential differentiating factors. Higher PreferredPropertyStar seems to correlate with ProdTaken.


### Multivariate NumberOfTrips vs. ProdTaken
![Multivariate NumberOfTrips vs. ProdTaken](box_NumberOfTrips_prod_taken.png)

**💡 Observation:** A comparison of **NumberOfTrips** distribution between buyers and non-buyers reveals potential differentiating factors. Higher NumberOfTrips seems to correlate with ProdTaken.


### Multivariate NumberOfChildrenVisiting vs. ProdTaken
![Multivariate NumberOfChildrenVisiting vs. ProdTaken](box_NumberOfChildrenVisiting_prod_taken.png)

**💡 Observation:** A comparison of **NumberOfChildrenVisiting** distribution between buyers and non-buyers reveals potential differentiating factors. Higher NumberOfChildrenVisiting seems to correlate with ProdTaken.


### Multivariate MonthlyIncome vs. ProdTaken
![Multivariate MonthlyIncome vs. ProdTaken](box_MonthlyIncome_prod_taken.png)

**💡 Observation:** A comparison of **MonthlyIncome** distribution between buyers and non-buyers reveals potential differentiating factors. Higher MonthlyIncome seems to correlate with ProdTaken.


### Multivariate PitchSatisfactionScore vs. ProdTaken
![Multivariate PitchSatisfactionScore vs. ProdTaken](box_PitchSatisfactionScore_prod_taken.png)

**💡 Observation:** A comparison of **PitchSatisfactionScore** distribution between buyers and non-buyers reveals potential differentiating factors. Higher PitchSatisfactionScore seems to correlate with ProdTaken.


### Multivariate TypeofContact vs. ProdTaken Proportions
![Multivariate TypeofContact vs. ProdTaken Proportions](stacked_bar_TypeofContact_prod_taken.png)

**💡 Observation:** This stacked bar chart shows the proportion of product taken within each category of **TypeofContact**, highlighting categories with higher purchase rates. Certain categories in TypeofContact show a significantly higher likelihood of purchase.


### Multivariate CityTier vs. ProdTaken Proportions
![Multivariate CityTier vs. ProdTaken Proportions](stacked_bar_CityTier_prod_taken.png)

**💡 Observation:** This stacked bar chart shows the proportion of product taken within each category of **CityTier**, highlighting categories with higher purchase rates. Certain categories in CityTier show a significantly higher likelihood of purchase.


### Multivariate Occupation vs. ProdTaken Proportions
![Multivariate Occupation vs. ProdTaken Proportions](stacked_bar_Occupation_prod_taken.png)

**💡 Observation:** This stacked bar chart shows the proportion of product taken within each category of **Occupation**, highlighting categories with higher purchase rates. Certain categories in Occupation show a significantly higher likelihood of purchase.


### Multivariate Gender vs. ProdTaken Proportions
![Multivariate Gender vs. ProdTaken Proportions](stacked_bar_Gender_prod_taken.png)

**💡 Observation:** This stacked bar chart shows the proportion of product taken within each category of **Gender**, highlighting categories with higher purchase rates. Certain categories in Gender show a significantly higher likelihood of purchase.


### Multivariate ProductPitched vs. ProdTaken Proportions
![Multivariate ProductPitched vs. ProdTaken Proportions](stacked_bar_ProductPitched_prod_taken.png)

**💡 Observation:** This stacked bar chart shows the proportion of product taken within each category of **ProductPitched**, highlighting categories with higher purchase rates. Certain categories in ProductPitched show a significantly higher likelihood of purchase.


### Multivariate MaritalStatus vs. ProdTaken Proportions
![Multivariate MaritalStatus vs. ProdTaken Proportions](stacked_bar_MaritalStatus_prod_taken.png)

**💡 Observation:** This stacked bar chart shows the proportion of product taken within each category of **MaritalStatus**, highlighting categories with higher purchase rates. Certain categories in MaritalStatus show a significantly higher likelihood of purchase.


### Multivariate Passport vs. ProdTaken Proportions
![Multivariate Passport vs. ProdTaken Proportions](stacked_bar_Passport_prod_taken.png)

**💡 Observation:** This stacked bar chart shows the proportion of product taken within each category of **Passport**, highlighting categories with higher purchase rates. Certain categories in Passport show a significantly higher likelihood of purchase.


### Multivariate OwnCar vs. ProdTaken Proportions
![Multivariate OwnCar vs. ProdTaken Proportions](stacked_bar_OwnCar_prod_taken.png)

**💡 Observation:** This stacked bar chart shows the proportion of product taken within each category of **OwnCar**, highlighting categories with higher purchase rates. Certain categories in OwnCar show a significantly higher likelihood of purchase.


### Multivariate Designation vs. ProdTaken Proportions
![Multivariate Designation vs. ProdTaken Proportions](stacked_bar_Designation_prod_taken.png)

**💡 Observation:** This stacked bar chart shows the proportion of product taken within each category of **Designation**, highlighting categories with higher purchase rates. Certain categories in Designation show a significantly higher likelihood of purchase.

