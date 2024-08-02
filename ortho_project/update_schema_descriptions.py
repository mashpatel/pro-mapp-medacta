import sqlite3

def read_descriptions(file_path):
    descriptions = {}
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.strip().split(': ', 1)
            if len(parts) == 2:
                attribute, description = parts
                descriptions[attribute] = description
    return descriptions

def update_schema_info(descriptions):
    conn = sqlite3.connect('Medacta.db')
    cursor = conn.cursor()

    # Get all attributes from schema_info
    cursor.execute("SELECT Attribute FROM Schema_Info")
    attributes = [row[0] for row in cursor.fetchall()]

    # Update descriptions for matching attributes
    for attribute in attributes:
        if attribute in descriptions:
            cursor.execute('''
            UPDATE Schema_Info
            SET Description = ?
            WHERE Attribute = ?
            ''', (descriptions[attribute], attribute))

    conn.commit()
    conn.close()

def main():
    try:
        file_path = 'Short_desc.txt'  # Path to your description file
        descriptions = read_descriptions(file_path)
        update_schema_info(descriptions)
        print("Schema descriptions have been updated successfully.")
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
    except sqlite3.Error as e:
        print(f"SQLite error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()