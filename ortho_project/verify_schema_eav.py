import sqlite3

# Connect to the SQLite database
conn = sqlite3.connect('Medacta.db')
cursor = conn.cursor()

# Retrieve all rows from the Schema_Info table
cursor.execute('SELECT * FROM Schema_Info')
rows = cursor.fetchall()

# Print the schema information
print("Schema Information:")
for row in rows:
    print(f"Attribute: {row[0]}, Description: {row[1]}")

# Close the connection
conn.close()