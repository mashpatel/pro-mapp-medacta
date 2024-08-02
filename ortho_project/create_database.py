

import pandas as pd
import sqlite3

# Read the spreadsheet
df = pd.read_excel('Medacta.Opnotes.And.Proms.xlsx')  # Use read_csv for CSV files

# Create a connection to the SQLite database
conn = sqlite3.connect('medacta.db')

# Write the data to a SQLite table
df.to_sql('Surgical_Data', conn, if_exists='replace', index=False)

# Close the connection
conn.close()

print("Database created successfully!")