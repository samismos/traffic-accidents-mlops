import pandas as pd
import mlrun
from sklearn.preprocessing import LabelEncoder

def date_parser(df, date_col, parsed_col, output_format):
    """
    Parses mixed-format date strings in a DataFrame column.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        date_col (str): The name of the column containing the original date strings.
        parsed_col (str): The name for the new column with parsed dates.
        output_format (str): The desired string format for the parsed dates (e.g., "%d-%m-%Y").

    Returns:
        pd.DataFrame: The DataFrame with an added column for parsed dates.
    """
    # Preserve the original date strings in a temporary column
    df['_original_' + date_col] = df[date_col]

    # List of expected date formats (order matters: try the most common first)
    formats = ["%d/%m/%Y", "%d-%m-%Y"]

    # First attempt: try the first format
    df[parsed_col] = pd.to_datetime(df['_original_' + date_col], format=formats[0], errors='coerce')
    
    # Identify rows where conversion failed
    mask = df[parsed_col].isna()
    
    # For the rows that failed, try the next formats
    for fmt in formats[1:]:
        if mask.sum() == 0:
            break  # All dates parsed; no need for further attempts
        df.loc[mask, parsed_col] = pd.to_datetime(
            df.loc[mask, '_original_' + date_col], format=fmt, errors='coerce'
        )
        # Update the mask for any remaining unparsed dates
        mask = df[parsed_col].isna()
    
    # Convert the parsed dates to a consistent string format
    df[parsed_col] = df[parsed_col].dt.strftime(output_format)
    
    # Replace the original date column with the parsed dates
    df[date_col] = df[parsed_col]

    # Drop the temporary columns
    df.drop(columns=['_original_' + date_col, parsed_col], inplace=True)

    # Extract useful features from the dates
    df['date'] = pd.to_datetime(df[date_col], dayfirst=True)
    df['day'] = df['date'].dt.day
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['day_of_week'] = df['date'].dt.dayofweek  # 0 = Monday, 6 = Sunday
    df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)

    df = df.drop(columns=['date', date_col])

    return df


def entrypoint(context: mlrun.MLClientCtx, dataset_uri, **args):
    # Ingest the dataset as dataframe
    df = dataset_uri.as_df()

    # Remove rows with empty values
    df = df.dropna()
    
    # Make all dates adhere to output_format
    df = date_parser(df, 'Accident Date', 'Parsed Date', "%d-%m-%Y")

    # Drop index
    df = df.drop(columns=['Index'])

    # Encode columns
    label_encoder = LabelEncoder()
    df['Accident_Severity'] = label_encoder.fit_transform(df['Accident_Severity'])

     # Print label encoding mapping
    for index, label in enumerate(label_encoder.classes_):
        print(f"Encoded label {index} corresponds to actual label: {label}")

    # Binary encoding
    df['Urban_or_Rural_Area'] = df['Urban_or_Rural_Area'].map({'Urban': 1, 'Rural': 0})

    # One-hot encoding
    categorical_cols = ['Light_Conditions', 'Road_Surface_Conditions', 'Road_Type', 
                         'Weather_Conditions', 'Vehicle_Type']
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    # Target encoding (if needed)
    mean_severity = df.groupby('District Area')['Accident_Severity'].transform('mean')
    df['District_Area_encoded'] = mean_severity
    df = df.drop(['District Area'], axis=1)


    # Save and log the processed DataFrame
    output_path = '/tmp/df.csv'
    df.to_csv(output_path, index=False)
    
    # Timestamp-based versioning
    version = args.get('VERSION')

    context.log_artifact(
        'processed_data',
        local_path=output_path,
        tag=version,
        labels={
            'rows': len(df)
        }
    )