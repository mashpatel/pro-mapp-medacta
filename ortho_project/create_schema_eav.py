import sqlite3
import csv

# Connect to the SQLite database
conn = sqlite3.connect('Medacta.db')
cursor = conn.cursor()

# Create a new table to store the schema
cursor.execute('''
CREATE TABLE IF NOT EXISTS Schema_Info (
    Attribute TEXT PRIMARY KEY,
    Description TEXT
)
''')

# Read the CSV file and insert data into the Schema_Info table
with open('schema_output.csv', 'r') as f:
    csv_reader = csv.reader(f)
    next(csv_reader)  # Skip the header row
    for row in csv_reader:
        cursor.execute('INSERT OR REPLACE INTO Schema_Info (Attribute, Description) VALUES (?, ?)', row)

# Commit the changes and close the connection
conn.commit()
conn.close()

print("Schema information has been loaded into the Schema_Info table.")