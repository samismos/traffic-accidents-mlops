import pandas as pd
import mlrun
from datetime import datetime

def date_parser(df, date_col, parsed_col, output_format):
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

    return df


def entrypoint(context: mlrun.MLClientCtx, dataset_uri, date_col='Accident Date', parsed_col='Parsed Date', output_format="%d-%m-%Y", **args):
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
    # Ingest the dataset as dataframe
    df = dataset_uri.as_df()
    
    # Count the original number of rows
    original_count = len(df)
    # Remove rows with empty values
    df = df.dropna()
    
    # Count the cleaned number of rows
    cleaned_count = len(df)

    # Calculate the number of deleted rows
    deleted_rows = original_count - cleaned_count

    print(f"Number of rows deleted: {deleted_rows}")
    print(f"Number of original rows: {original_count}")
    print(f"Number of clean rows: {cleaned_count}")



    # Make all dates adhere to output_format
    df = date_parser(df, date_col, parsed_col, output_format)

    # Save and log the processed DataFrame
    output_path = '/tmp/df.csv'
    df.to_csv(output_path, index=False)
    
    # Timestamp-based versioning
    version = args.get('VERSION')

    context.log_artifact(
        f'processed_data_{version}',
        local_path=output_path,
    )