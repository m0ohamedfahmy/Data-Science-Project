# Import the main Libraries
import pandas as pd
import numpy as np
##  preprocessing
from sklearn.model_selection import train_test_split , cross_val_score, StratifiedKFold,cross_val_predict
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.impute import KNNImputer ,SimpleImputer
from sklearn.preprocessing import PowerTransformer, OneHotEncoder, OrdinalEncoder, MinMaxScaler
from sklearn_features.transformers import DataFrameSelector
from datasist.structdata import detect_outliers


path_data = r"D:\Software_Courses\Coding\Data Science Project\Supervised\Classification\Obesity based on eating habits and physical conditions\data\raw/ObesityDataSet.csv"

# Load data  and print first 5 sample
df = pd.read_csv(path_data)


# Each column name converting them to lowercase
df.columns = df.columns.str.lower()

## Round to two decimal place all Numeircal columns
df[df.select_dtypes(exclude="object").columns] = df.select_dtypes(
    exclude="object"
).apply(lambda x: round(x, 2))

#### Drop duplicates rows
df.drop_duplicates(inplace=True)

# Create new feature (BMI)
df["bmi"] = round(df.weight / np.square(df.height), 1)

# Calculate number of outlier for each columns and git index of outlier
outlier_index = []
for i in df.select_dtypes(exclude="object").columns:
    if len(detect_outliers(df, 0, [i])) == 0:
        continue
    else:
        outlier_index.extend(detect_outliers(df, 0, [i]))

## Nature Data
df_nature = df.iloc[~df.index.isin(outlier_index)]


# Split dataset into X & Y
# Drop specified columns ('nobeyesdad', 'gender', 'age', 'height', 'weight', 'smoke') to create the feature matrix (X)
# 'x' will contain the features, and 'y' will contain the target variable
x = df_nature.drop(columns=['nobeyesdad', 'gender', 'age', 'height', 'weight', 'smoke'])

# 'y' will be the target variable, which is 'nobeyesdad' in this case
y = df_nature['nobeyesdad']

# Split dataset into training and testing sets
# 'x_train' and 'y_train' will contain the training data and labels
# 'x_test' and 'y_test' will contain the testing data and labels
# The data will be split into a 75% training set and a 25% testing set, with shuffling and a random seed for reproducibility
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, shuffle=True, random_state=42, stratify=y)



# Get the names of numerical columns (exclude "object" dtype columns)
numircal_col = x_train.select_dtypes(exclude="object").columns.tolist()

# Get the names of ordinal categorical columns
catego_col_ordinal = x_train[["caec", "calc", "mtrans"]].columns.tolist()

# Get the names of nominal categorical columns
catego_col_nominal = x_train[["family_history_with_overweight", "favc", "scc"]].columns.tolist()



## Create PipeLine

#### Numrical Pipeline
num_pipe = Pipeline(
    steps=[
        ("selector", DataFrameSelector(numircal_col)),
        ("imputer", KNNImputer()),
        ("box-cox", PowerTransformer(method="yeo-johnson", standardize=False)),
        ("normalization", MinMaxScaler(feature_range=(-1, 1))),
    ]
)

## Categorical(ordinal) Pipline
ordinal_pipe = Pipeline(
    steps=[
        ("selector", DataFrameSelector(catego_col_ordinal)),
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoding", OrdinalEncoder()),
    ]
)

## Categorical(nominal) Pipline
nominal_pipe = Pipeline(
    steps=[
        ("selector", DataFrameSelector(catego_col_nominal)),
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoding", OneHotEncoder(drop="first", sparse_output=False)),
    ]
)

## Combine all
all_pipe = FeatureUnion(
    transformer_list=[
        ("Numrical", num_pipe),
        ("ordinal", ordinal_pipe),
        ("nominal", nominal_pipe),
    ]
)

## Apply
_ = all_pipe.fit(x_train)


def process_now(x_new):
    # creat data frame
    df_new=pd.DataFrame([x_new],columns=['family_history_with_overweight','favc','fcvc','ncp','caec','ch2o','scc','faf','tue','calc','mtrans','height','weight'])


    # Adjust the data types 
    df_new['family_history_with_overweight']=df_new['family_history_with_overweight'].astype('str')
    df_new['favc']=df_new['favc'].astype('str')
    df_new['fcvc']=df_new['fcvc'].astype('float')
    df_new['ncp']=df_new['ncp'].astype('float')
    df_new['caec']=df_new['caec'].astype('str')
    df_new['ch2o']=df_new['ch2o'].astype('float')
    df_new['scc']=df_new['scc'].astype('str')
    df_new['faf']=df_new['faf'].astype('float')
    df_new['tue']=df_new['tue'].astype('float')
    df_new['calc']=df_new['calc'].astype('str')
    df_new['mtrans']=df_new['mtrans'].astype('str')
    df_new['height']=df_new['height'].astype('float')
    df_new['weight']=df_new['weight'].astype('float')
    
    # Feature Eng
    df_new["bmi"] = round(df_new.weight / np.square(df_new.height), 1)
    df_new['bmi']=df_new['bmi'].astype('float')

    ## apply the pipeline
    x_processed = all_pipe.transform(df_new)

    return x_processed