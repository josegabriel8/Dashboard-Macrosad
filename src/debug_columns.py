import pandas as pd

# Load the data
df = pd.read_excel('expedientes_processed.xlsx')

print('Available columns:')
for i, col in enumerate(df.columns):
    print(f'{i}: {col}')

print(f'\nestado_usuario exists: {"estado_usuario" in df.columns}')

if 'estado_usuario' in df.columns:
    print(f'Unique values in estado_usuario: {df["estado_usuario"].unique()}')
    print(f'Value counts: {df["estado_usuario"].value_counts()}')
else:
    # Look for similar column names
    similar = [col for col in df.columns if 'estado' in col.lower() or 'user' in col.lower()]
    print(f'Similar columns: {similar}')
