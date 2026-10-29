import mysql.connector

try:
    connection = mysql.connector.connect(
        host="127.0.0.1",
        port=3306,
        user="root",
        password="",
        database="driver_drowsiness"
    )

    if connection.is_connected():
        print("====================================")
        print("MYSQL CONNECTION SUCCESSFUL")
        print("Database: driver_drowsiness")
        print("====================================")

        cursor = connection.cursor()

        cursor.execute("SHOW TABLES")

        print("\nTables found:")

        for table in cursor.fetchall():
            print("-", table[0])

        cursor.close()
        connection.close()

except mysql.connector.Error as error:

    print("====================================")
    print("MYSQL CONNECTION FAILED")
    print("====================================")
    print("Error:", error)