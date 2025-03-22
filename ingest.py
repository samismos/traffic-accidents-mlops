# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mlrun
# import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE
# Classification models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

def entrypoint(context: mlrun.MLClientCtx, DATASET_URI, **args):
    # For reproducibility
    np.random.seed(42)

    # =====================
    # 1. Data Loading
    # =====================
    # Replace 'uk_traffic_accidents.csv' with your dataset file path.
    # data = pd.read_csv('LONDON.csv')
    data = DATASET_URI.as_df()

    # =====================
    # 2. Initial Data Exploration
    # =====================
    print("Dataset shape:", data.shape)
    print("\nDataset Info:")
    print(data.info())
    print("\nMissing values:")
    print(data.isnull().sum())
    print("\nDescriptive statistics:")
    print(data.describe())

    # =====================
    # 3. Exploratory Data Analysis (EDA)
    # =====================

    # Plotting distribution of target variable: Accident Severity
    # plt.figure(figsize=(8, 5))
    # sns.countplot(x='Accident_Severity', data=data, palette='viridis')
    # plt.title("Distribution of Accident Severity")
    # plt.xlabel("Accident Severity")
    # plt.ylabel("Count")
    # plt.show()

    # If there are categorical variables other than the target, explore them too
    categorical_cols = data.select_dtypes(include=['object']).columns
    print("\nCategorical Columns:", categorical_cols)

    # Plot counts for each categorical feature (limit to a few for clarity)
    # for col in categorical_cols:
    #     plt.figure(figsize=(8, 4))
    #     sns.countplot(y=col, data=data, order=data[col].value_counts().index, palette='magma')
    #     plt.title(f"Distribution of {col}")
    #     plt.xlabel("Count")
    #     plt.ylabel(col)
    #     plt.show()

    # Correlation heatmap for numerical features
    numerical_cols = data.select_dtypes(include=['int64', 'float64']).columns
    # plt.figure(figsize=(10, 8))
    # sns.heatmap(data[numerical_cols].corr(), annot=True, cmap='coolwarm', fmt=".2f")
    # plt.title("Correlation Heatmap of Numerical Features")
    # plt.show()

    # =====================
    # 4. Data Preprocessing
    # =====================

    # Handling missing values:
    # Fill missing numerical values with the median and categorical ones with the mode.
    for col in data.columns:
        if data[col].dtype in ['int64', 'float64']:
            data[col].fillna(data[col].median(), inplace=True)
        else:
            data[col].fillna(data[col].mode()[0], inplace=True)

    # Converting categorical variables to numeric using Label Encoding
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
        label_encoders[col] = le

    # Optional: Scale numerical features if needed (excluding the target variable)
    scaler = StandardScaler()
    features_to_scale = [col for col in numerical_cols if col != 'Accident Severity']
    data[features_to_scale] = scaler.fit_transform(data[features_to_scale])

    # =====================
    # 5. Prepare Data for Modeling
    # =====================

    print(data.columns)

    # Define features (X) and target (y)
    X = data.drop('Accident_Severity', axis=1)
    y = data['Accident_Severity']

    # Drop Index column
    # X = X.drop(columns=['Index'])


    # Split data into training and testing sets using stratification to maintain class proportions
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    print("\nTraining set shape:", X_train.shape)
    print("Testing set shape:", X_test.shape)

    # =====================
    # 6. Handle Class Imbalance with SMOTE
    # =====================

    # Print original class distribution in training set
    print("\nOriginal training set class distribution:")
    print(y_train.value_counts())

    # Define a sampling strategy for the minority classes (adjust the labels and target counts as needed)
    # sampling_strategy = {1: 2000, 2: 2000}  # Replace 1 and 2 with the actual minority class labels

    # Create the SMOTE object with the desired sampling strategy
    smote = SMOTE(sampling_strategy='auto', random_state=42)

    # Apply SMOTE to the training data
    X_train_over, y_train_over = smote.fit_resample(X_train, y_train)

    print("Shapes before saving:")
    print(f"X_train_over: {X_train_over.shape}, y_train_over: {y_train_over.shape}")
    print(f"X_test: {X_test.shape}, y_test: {y_test.shape}")

    print("\nAfter SMOTE, training set class distribution:")
    print(y_train_over.value_counts())

    X_train_over_file = '/tmp/X_train_over.csv'
    y_train_over_file = '/tmp/y_train_over.csv'
    X_test_file = '/tmp/X_test.csv'
    y_test_file = '/tmp/y_test.csv'
    X_train_over.to_csv(X_train_over_file, index=False)
    y_train_over.to_csv(y_train_over_file, index=False)
    X_test.to_csv(X_test_file, index=False)
    y_test.to_csv(y_test_file, index=False)

    print("Shapes after loading:")
    print(f"X_train_over: {X_train_over.shape}, y_train_over: {y_train_over.shape}")
    print(f"X_test: {X_test.shape}, y_test: {y_test.shape}")

    # Timestamp-based versioning
    version = args.get('VERSION')
    
    artifact_prefix = "B_" if args.get("IS_BATCH") else ""
    
    context.log_artifact(
        f'{artifact_prefix}X_train',
        local_path=X_train_over_file,
        tag=version,
        # labels={'rows': len(X_train_over),'columns': len(X_train_over.columns)}
    )

    context.log_artifact(
        f'{artifact_prefix}y_train',
        local_path=y_train_over_file,
        tag=version,
        # labels={'rows': len(y_train_over),'columns': len(y_train_over.columns)}
    )

    context.log_artifact(
        f'{artifact_prefix}X_test',
        local_path=X_test_file,
        tag=version,
        # labels={'rows': len(X_test),'columns': len(X_test.columns)}
    )

    context.log_artifact(
        f'{artifact_prefix}y_test',
        local_path=y_test_file,
        tag=version,
        # labels={'rows': len(y_test),'columns': len(y_test.columns)}
    )