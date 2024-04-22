import os
import psycopg2
import psycopg2.extras
from dotenv import load_dotenv

def initialize_conn():
    # Load environment variables from .env file
    load_dotenv()

    db_url = os.getenv("DATABASE_URL")
    # Get database connection details from environment variables
    print("Trying to connect to the database")
    conn = psycopg2.connect(db_url,
                                application_name="$ docs_simplecrud_psycopg2", 
                                cursor_factory=psycopg2.extras.RealDictCursor)
    print("Connected to the database")

    psycopg2.extras.register_uuid()
    return conn

def close_conn(conn):
    # Close communication with the database
    conn.close()
    print("Closed the connection to the database")


def upsert_node_measurements(conn, node_measurements):
    # Create a cursor object to execute SQL queries
    cursor = conn.cursor()

    # Upsert the Node_Measurements rows
    for index, row in node_measurements.iterrows():
        cursor.execute("""
            UPSERT INTO Node_Measurement (uuid, node_id, timestamp, temperature, pH, dissolved_oxygen)
            VALUES (%s, %s, %s, %s, %s, %s)
        """, (row['uuid'], row['node_id'], row['timestamp'], row['temperature'], row['pH'], row['dissolved_oxygen']))

    # Commit the changes to the database
    conn.commit()

def select_node_measurements(conn):
    # Create a cursor object to execute SQL queries
    cursor = conn.cursor()

    # Select all the Node_Measurement rows
    cursor.execute("SELECT * FROM Node_Measurement")

    # Fetch all the rows from the result
    rows = cursor.fetchall()

    # Commit the changes to the database
    conn.commit()

    return rows

def delete_all_node_measurements(conn):
    # Create a cursor object to execute SQL queries
    cursor = conn.cursor()
    # Delete all the Node_Measurement rows
    cursor.execute("DELETE FROM Node_Measurement")
    # Commit the changes to the database
    conn.commit()
    print("Deleted all node measurements")