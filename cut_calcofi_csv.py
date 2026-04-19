import pandas as pd

def extract_clean_phytoplankton_data(input_file, output_file):
    # The final set of 10 parameters
    target_columns = [
        'Date','Lat_Dec', 'Lon_Dec', 'T_degC', 'Salnty', 
        'O2ml_L', 'O2Sat', 'ChlorA', 'Phaeop', 
        'Si03uM', 'NO3uM'
    ]
    
    try:
        df = pd.read_csv(input_file, low_memory=False)

        df.columns = df.columns.str.strip()
        
        available_cols = [col for col in target_columns if col in df.columns]
        
        extracted_df = df[available_cols]
        
        clean_df = extracted_df.dropna(how='any')
        
        clean_df.to_csv(output_file, index=False)
        
        print("--- Extraction Successful ---")
        print(f"Original rows: {len(df)}")
        print(f"Final clean rows: {len(clean_df)}")
        print(f"Rows removed due to missing data: {len(df) - len(clean_df)}")
        print(f"File saved as: {output_file}")

    except Exception as e:
        print(f"An error occurred: {e}")

extract_clean_phytoplankton_data('Cleaned_CalCOFI_Phytoplankton.csv', 'Finalized_CalCOFI_Phytoplankton.csv')