# ==========================================================
# db.py
# AI DRIVER DROWSINESS DETECTION SYSTEM
# MYSQL DATABASE CONNECTION
# ==========================================================

import mysql.connector
from mysql.connector import Error


# ==========================================================
# MYSQL CONFIGURATION
# ==========================================================

DB_CONFIG = {
    "host": "127.0.0.1",
    "port": 3306,
    "user": "root",
    "password": "",
    "database": "driver_drowsiness"
}


# ==========================================================
# GET MYSQL CONNECTION
# ==========================================================

def get_connection():

    try:

        connection = mysql.connector.connect(
            host=DB_CONFIG["host"],
            port=DB_CONFIG["port"],
            user=DB_CONFIG["user"],
            password=DB_CONFIG["password"],
            database=DB_CONFIG["database"]
        )

        if connection.is_connected():
            return connection

    except Error as error:

        print("MySQL connection error:", error)

    return None


# ==========================================================
# INITIALIZE DATABASE
# ==========================================================

def init_db():

    connection = None
    cursor = None

    try:

        connection = mysql.connector.connect(
            host=DB_CONFIG["host"],
            port=DB_CONFIG["port"],
            user=DB_CONFIG["user"],
            password=DB_CONFIG["password"]
        )

        cursor = connection.cursor()

        # --------------------------------------------------
        # Create Database
        # --------------------------------------------------

        cursor.execute("""
            CREATE DATABASE IF NOT EXISTS driver_drowsiness
        """)

        connection.commit()

        cursor.close()
        connection.close()

        # --------------------------------------------------
        # Connect to project database
        # --------------------------------------------------

        connection = get_connection()

        if connection is None:
            raise Exception(
                "Could not connect to driver_drowsiness database."
            )

        cursor = connection.cursor()

        # ==================================================
        # USERS TABLE
        # ==================================================

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (

                id INT AUTO_INCREMENT PRIMARY KEY,

                username VARCHAR(100)
                NOT NULL UNIQUE,

                email VARCHAR(150)
                NOT NULL UNIQUE,

                password VARCHAR(255)
                NOT NULL,

                full_name VARCHAR(150)
                DEFAULT '',

                phone_number VARCHAR(30)
                DEFAULT '',

                date_of_birth VARCHAR(30)
                DEFAULT '',

                gender VARCHAR(30)
                DEFAULT '',

                licence_id VARCHAR(100)
                DEFAULT '',

                licence_type VARCHAR(100)
                DEFAULT '',

                licence_issue_date VARCHAR(30)
                DEFAULT '',

                licence_expiry_date VARCHAR(30)
                DEFAULT '',

                vehicle_number VARCHAR(50)
                DEFAULT '',

                experience INT
                DEFAULT 0,

                registration_date DATETIME
                DEFAULT CURRENT_TIMESTAMP
            )
        """)


        # ==================================================
        # FACE SAMPLES TABLE
        # ==================================================

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS face_samples (

                id INT AUTO_INCREMENT PRIMARY KEY,

                user_id INT NOT NULL,

                image LONGBLOB NOT NULL,

                created_at TIMESTAMP
                DEFAULT CURRENT_TIMESTAMP,

                CONSTRAINT fk_face_user
                FOREIGN KEY (user_id)
                REFERENCES users(id)
                ON DELETE CASCADE
            )
        """)


        # ==================================================
        # REPORTS TABLE
        # ==================================================

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS reports (

                id INT AUTO_INCREMENT PRIMARY KEY,

                user_id INT NULL,

                username VARCHAR(100),

                status VARCHAR(100),

                created_at DATETIME
                DEFAULT CURRENT_TIMESTAMP,

                CONSTRAINT fk_report_user
                FOREIGN KEY (user_id)
                REFERENCES users(id)
                ON DELETE CASCADE
            )
        """)


        connection.commit()

        print("")
        print("========================================")
        print("MYSQL DATABASE INITIALIZED SUCCESSFULLY")
        print("========================================")
        print("Database : driver_drowsiness")
        print("Tables   : users")
        print("           face_samples")
        print("           reports")
        print("========================================")


    except Error as error:

        print("")
        print("========================================")
        print("MYSQL DATABASE ERROR")
        print("========================================")
        print(error)
        print("========================================")


    except Exception as error:

        print("Database initialization error:", error)


    finally:

        if cursor:

            try:
                cursor.close()
            except Exception:
                pass

        if connection:

            try:
                connection.close()
            except Exception:
                pass


# ==========================================================
# TEST DATABASE CONNECTION
# ==========================================================

def test_connection():

    connection = None
    cursor = None

    try:

        connection = get_connection()

        if connection is None:

            print("MYSQL CONNECTION FAILED")
            return False

        cursor = connection.cursor()

        cursor.execute("""
            SELECT DATABASE()
        """)

        result = cursor.fetchone()

        print("")
        print("========================================")
        print("MYSQL CONNECTION SUCCESSFUL")
        print("========================================")
        print("Database:", result[0])

        cursor.execute("""
            SHOW TABLES
        """)

        tables = cursor.fetchall()

        print("")
        print("Tables found:")

        for table in tables:

            print("-", table[0])

        print("========================================")

        return True


    except Exception as error:

        print("MySQL test error:", error)

        return False


    finally:

        if cursor:

            try:
                cursor.close()
            except Exception:
                pass

        if connection:

            try:
                connection.close()
            except Exception:
                pass


# ==========================================================
# RUN DB TEST
# ==========================================================

if __name__ == "__main__":

    print("")
    print("Testing MySQL connection...")
    print("")

    init_db()

    test_connection()