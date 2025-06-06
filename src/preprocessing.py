import pandas as pd

def load_and_clean_data(file_path):
    # Load the Excel file
    df = pd.read_excel(file_path, engine='openpyxl')

    # Convert numeric fields
    numeric_fields = ['edad', 'horas_pia', 'horas_dom', 'horas_per']
    for field in numeric_fields:
        df[field] = pd.to_numeric(df[field], errors='coerce')

    # Convert other categoricals to category type
    categorical_fields = ['sexo_u', 'c_post_u', 'tipo_servicio']
    for field in categorical_fields:
        if field in df.columns:
            df[field] = df[field].astype('category')

    return df

def preprocess_and_save_data(input_file, output_file):
    # Load the Excel file
    df = pd.read_excel(input_file, engine='openpyxl')

    # Rename columns to snake_case

    # Convert numeric fields
    numeric_fields = ['edad', 'horas_pia', 'horas_dom', 'horas_per']
    for field in numeric_fields:
        df[field] = pd.to_numeric(df[field], errors='coerce')

    # Clean grado_dep field
    if 'grado_dep' in df.columns:
        df['grado_dep'] = df['grado_dep'].str.replace('Grado ', '', regex=False).astype(str)

    # Convert other categoricals to category type
    categorical_fields = ['sexo_u', 'c_post_u', 'tipo_servicio']
    for field in categorical_fields:
        if field in df.columns:
            df[field] = df[field].astype('category')

    # Bin horas_pia into 5 intervals
    if 'horas_pia' in df.columns:
        df['horas_pia_interval'] = pd.cut(df['horas_pia'], bins=5, labels=False)

    # Save the preprocessed data to a new Excel file
    df.to_excel(output_file, index=False, engine='openpyxl')

# Example usage
if __name__ == "__main__":
    input_file = 'src/db_usuarios.xlsx'
    output_file = 'src/db_usuarios_preprocessed.xlsx'
    preprocess_and_save_data(input_file, output_file)

