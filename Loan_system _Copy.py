# import data and cleaning
import pandas as pd
from sklearn.model_selection import train_test_split
df = pd.read_csv('loan_approval_data.csv')
print(df.head)

print('number of null values :', df.isnull().sum())
print('number of duplicate values :', df.duplicated().sum())

#separate categorical and numerical colums:
categorical_cols = df.select_dtypes(include=['object']).columns
numerical_cols = df.select_dtypes(include=['float64']).columns

#handling missing values using imputer
from sklearn.impute import SimpleImputer

num_imp = SimpleImputer(strategy = 'mean')
df[numerical_cols] = num_imp.fit_transform(df[numerical_cols])

cat_imp = SimpleImputer(strategy = 'most_frequent')
df[categorical_cols] = cat_imp.fit_transform(df[categorical_cols])


print('after number of null values :', df.isnull().sum())

# Exploratory Data Analysis
import matplotlib.pyplot as plt
import seaborn as sns
classes_count = df['Loan_Approved'].value_counts()
plt.pie(classes_count , labels = ['No', 'Yes'], autopct = '%1.1f%%')
plt.title('Loan Approval Distribution')


#analyze categories

gender_cnt = df['Gender'].value_counts()
sns.barplot(gender_cnt)


#analyze income 

sns.histplot(
    data = df,
    x = 'Applicant_Income',
    bins = 20
)

plt.show()

# outliers - box plot
sns.boxplot(
    data = df,
    x = 'Loan_Approved',
    y = 'Applicant_Income'
)
plt.show()

# remove applicant Id 
df = df.drop('Applicant_ID', axis = 1)

#Feature Encoding (using sklearn encoder)

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

le = LabelEncoder()
df['Education_Level'] = le.fit_transform(df['Education_Level'])
df['Loan_Approved'] = le.fit_transform(df['Loan_Approved'])


col = ['Employment_Status','Marital_Status','Loan_Purpose','Property_Area','Gender','Employer_Category']
ohe = OneHotEncoder(drop = 'first' , sparse_output = False, handle_unknown = 'ignore')
encoded_cols = ohe.fit_transform(df[col])

encoded_df =  pd.DataFrame(encoded_cols ,columns= ohe.get_feature_names_out(col), index = df.index)

df =pd.concat([df.drop(columns = col ) , encoded_df] , axis = 1)
print('After Encoding',df.head())
print('After Encoding',df.describe())

#Correlation Heatmap

nums_cols = df.select_dtypes(include = 'number')
corr_matrix = nums_cols.corr()

sns.heatmap(
    corr_matrix,
    annot = True,
    fmt = '.2f',
    cmap = 'coolwarm'
)

plt.show()
print('Correlation Matrix :', corr_matrix)
print('Correlation Matrix :\n', nums_cols.corr()['Loan_Approved'].sort_values(ascending=False))

#train_test_split
from sklearn.model_selection import train_test_split
X = df.drop('Loan_Approved',axis = 1)
y = df['Loan_Approved']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

#feature_scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#train and evaluate models

#logidtic regression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
log_model = LogisticRegression()
log_model.fit(X_train_scaled, y_train)

y_pred = log_model.predict(X_test_scaled)

print(f"""
Logistic Regression Model Evaluation:
------------------------------------
Confusion Matrix:
{confusion_matrix(y_test, y_pred)}

Classification Report:
{classification_report(y_test, y_pred)}

Metrics Summary:
* Accuracy Score  : {accuracy_score(y_test, y_pred):.4f}
* Precision Score : {precision_score(y_test, y_pred):.4f}
* Recall Score    : {recall_score(y_test, y_pred):.4f}
* F1 Score        : {f1_score(y_test, y_pred):.4f}
""")


#KNN
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train_scaled, y_train)

y_pred = knn_model.predict(X_test_scaled)

print(f"""
K-Nearest Neighbors Model Evaluation:
------------------------------------
Confusion Matrix:
{confusion_matrix(y_test, y_pred)}

Classification Report:
{classification_report(y_test, y_pred)}

Metrics Summary:
* Accuracy Score  : {accuracy_score(y_test, y_pred):.4f}
* Precision Score : {precision_score(y_test, y_pred):.4f}
* Recall Score    : {recall_score(y_test, y_pred):.4f}
* F1 Score        : {f1_score(y_test, y_pred):.4f}
""")

#Naive Bayes
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
nb_model = GaussianNB()
nb_model.fit(X_train_scaled, y_train)

y_pred = nb_model.predict(X_test_scaled)

print(f"""
Naive Bayes Model Evaluation:
------------------------------------
Confusion Matrix:
{confusion_matrix(y_test, y_pred)}

Classification Report:
{classification_report(y_test, y_pred)}

Metrics Summary:
* Accuracy Score  : {accuracy_score(y_test, y_pred):.4f}
* Precision Score : {precision_score(y_test, y_pred):.4f}
* Recall Score    : {recall_score(y_test, y_pred):.4f}
* F1 Score        : {f1_score(y_test, y_pred):.4f}
""")

#feature Engineering to improve model performance
df 