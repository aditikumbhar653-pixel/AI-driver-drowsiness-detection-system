import os
import base64
from functools import wraps
from datetime import datetime

import cv2
import numpy as np

from flask import (
    Flask,
    Response,
    render_template,
    request,
    redirect,
    url_for,
    session,
    flash,
    jsonify,
    g
)

from werkzeug.security import (
    generate_password_hash,
    check_password_hash
)

from detector import generate_frames
from db import get_connection, init_db


# ==========================================================
# 1. FLASK CONFIGURATION
# ==========================================================

app = Flask(__name__)

app.secret_key = os.environ.get(
    "FLASK_SECRET_KEY",
    "driver_drowsiness_project_secret_key"
)

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

FACE_MODEL_PATH = os.path.join(
    BASE_DIR,
    "face_model.yml"
)

REQUIRED_SAMPLES = 20


# ==========================================================
# 2. FACE CASCADE
# ==========================================================

CASCADE_PATH = os.path.join(
    cv2.data.haarcascades,
    "haarcascade_frontalface_default.xml"
)

face_cascade = cv2.CascadeClassifier(
    CASCADE_PATH
)

if face_cascade.empty():
    raise RuntimeError(
        f"Face cascade file could not be loaded: {CASCADE_PATH}"
    )


# ==========================================================
# 3. MYSQL DATABASE CONNECTION
# ==========================================================

def get_db():

    if "db" not in g:

        g.db = get_connection()

        if g.db is None:
            raise RuntimeError(
                "Could not connect to MySQL database."
            )

    return g.db


@app.teardown_appcontext
def close_db(error=None):

    db = g.pop("db", None)

    if db is not None:

        try:
            db.close()

        except Exception:
            pass


# ==========================================================
# 4. INITIALIZE DATABASE
# ==========================================================

try:

    init_db()

    print("----------------------------------------")
    print("MYSQL DATABASE INITIALIZED")
    print("----------------------------------------")

except Exception as error:

    print(
        "MySQL initialization error:",
        error
    )


# ==========================================================
# 5. ENSURE MYSQL DATABASE SCHEMA
# ==========================================================

def ensure_mysql_schema():

    db = None
    cursor = None

    try:

        db = get_connection()

        if db is None:
            return

        cursor = db.cursor()

        # --------------------------------------------------
        # USERS TABLE
        # --------------------------------------------------

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (

                id INT AUTO_INCREMENT PRIMARY KEY,

                username VARCHAR(100)
                NOT NULL UNIQUE,

                email VARCHAR(150)
                NOT NULL UNIQUE,

                password VARCHAR(255)
                NOT NULL

            )
        """)

        cursor.execute("""
            SELECT COLUMN_NAME
            FROM INFORMATION_SCHEMA.COLUMNS
            WHERE TABLE_SCHEMA = DATABASE()
            AND TABLE_NAME = 'users'
        """)

        user_columns = {
            row[0]
            for row in cursor.fetchall()
        }

        user_extra_columns = {

            "full_name":
                "VARCHAR(150) DEFAULT ''",

            "phone_number":
                "VARCHAR(30) DEFAULT ''",

            "date_of_birth":
                "VARCHAR(30) DEFAULT ''",

            "gender":
                "VARCHAR(30) DEFAULT ''",

            "licence_id":
                "VARCHAR(100) DEFAULT ''",

            "licence_type":
                "VARCHAR(100) DEFAULT ''",

            "licence_issue_date":
                "VARCHAR(30) DEFAULT ''",

            "licence_expiry_date":
                "VARCHAR(30) DEFAULT ''",

            "vehicle_number":
                "VARCHAR(50) DEFAULT ''",

            "experience":
                "INT DEFAULT 0",

            "registration_date":
                "DATETIME DEFAULT CURRENT_TIMESTAMP"
        }

        for column, definition in user_extra_columns.items():

            if column not in user_columns:

                cursor.execute(
                    f"""
                    ALTER TABLE users
                    ADD COLUMN {column} {definition}
                    """
                )

        # --------------------------------------------------
        # FACE SAMPLES TABLE
        # --------------------------------------------------

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS face_samples (

                id INT AUTO_INCREMENT PRIMARY KEY,

                user_id INT NOT NULL,

                image LONGBLOB NOT NULL,

                created_at TIMESTAMP
                DEFAULT CURRENT_TIMESTAMP

            )
        """)

        # --------------------------------------------------
        # REPORTS TABLE
        # --------------------------------------------------

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS reports (

                id INT AUTO_INCREMENT PRIMARY KEY,

                user_id INT NULL,

                username VARCHAR(100),

                status VARCHAR(100),

                created_at DATETIME
                DEFAULT CURRENT_TIMESTAMP

            )
        """)

        # --------------------------------------------------
        # CHECK REPORT COLUMNS
        # --------------------------------------------------

        cursor.execute("""
            SELECT COLUMN_NAME
            FROM INFORMATION_SCHEMA.COLUMNS
            WHERE TABLE_SCHEMA = DATABASE()
            AND TABLE_NAME = 'reports'
        """)

        report_columns = {
            row[0]
            for row in cursor.fetchall()
        }

        if "user_id" not in report_columns:

            cursor.execute("""
                ALTER TABLE reports
                ADD COLUMN user_id INT NULL
            """)

        if "username" not in report_columns:

            cursor.execute("""
                ALTER TABLE reports
                ADD COLUMN username VARCHAR(100)
            """)

        if "status" not in report_columns:

            cursor.execute("""
                ALTER TABLE reports
                ADD COLUMN status VARCHAR(100)
            """)

        if "created_at" not in report_columns:

            cursor.execute("""
                ALTER TABLE reports
                ADD COLUMN created_at DATETIME
                DEFAULT CURRENT_TIMESTAMP
            """)

        db.commit()

        print(
            "MySQL schema checked successfully."
        )

    except Exception as error:

        print(
            "Schema update error:",
            error
        )

        if db:

            try:
                db.rollback()

            except Exception:
                pass

    finally:

        if cursor:

            try:
                cursor.close()

            except Exception:
                pass

        if db:

            try:
                db.close()

            except Exception:
                pass


try:

    ensure_mysql_schema()

except Exception as error:

    print(
        "Schema initialization failed:",
        error
    )


# ==========================================================
# 6. LOGIN REQUIRED DECORATOR
# ==========================================================

def login_required(route_function):

    @wraps(route_function)
    def wrapper(*args, **kwargs):

        if "user_id" not in session:

            flash(
                "Please login first.",
                "error"
            )

            return redirect(
                url_for("login")
            )

        return route_function(
            *args,
            **kwargs
        )

    return wrapper


# ==========================================================
# 7. FACE DETECTION AND FACE CROP
# ==========================================================

def get_face_crop(image):

    if image is None or image.size == 0:
        return None

    if len(image.shape) == 3:

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

    else:

        gray = image.copy()

    gray = cv2.equalizeHist(gray)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60)
    )

    if len(faces) == 0:
        return None

    x, y, w, h = max(
        faces,
        key=lambda face:
        face[2] * face[3]
    )

    margin = int(
        0.15 * max(w, h)
    )

    x1 = max(
        0,
        x - margin
    )

    y1 = max(
        0,
        y - margin
    )

    x2 = min(
        gray.shape[1],
        x + w + margin
    )

    y2 = min(
        gray.shape[0],
        y + h + margin
    )

    face_crop = gray[
        y1:y2,
        x1:x2
    ]

    if face_crop.size == 0:
        return None

    return cv2.resize(
        face_crop,
        (200, 200)
    )


# ==========================================================
# 8. TRAIN LBPH FACE MODEL
# ==========================================================

def train_face_model():

    if not hasattr(cv2, "face"):

        print(
            "ERROR: opencv-contrib-python is required."
        )

        return False

    db = None
    cursor = None

    try:

        db = get_connection()

        if db is None:

            print(
                "Could not connect to MySQL."
            )

            return False

        cursor = db.cursor()

        cursor.execute("""
            SELECT
                fs.user_id,
                fs.image

            FROM face_samples AS fs

            INNER JOIN users AS u
                ON u.id = fs.user_id

            ORDER BY
                fs.user_id,
                fs.id
        """)

        rows = cursor.fetchall()

        training_images = []
        labels = []

        for user_id, image_bytes in rows:

            try:

                image_array = np.frombuffer(
                    image_bytes,
                    dtype=np.uint8
                )

                face_image = cv2.imdecode(
                    image_array,
                    cv2.IMREAD_GRAYSCALE
                )

                if (
                    face_image is None
                    or face_image.size == 0
                ):
                    continue

                face_image = cv2.resize(
                    face_image,
                    (200, 200)
                )

                training_images.append(
                    face_image
                )

                labels.append(
                    int(user_id)
                )

            except Exception as error:

                print(
                    "Training sample error:",
                    error
                )

        print(
            "Total training images:",
            len(training_images)
        )

        if len(training_images) < REQUIRED_SAMPLES:

            print(
                "Not enough face samples."
            )

            return False

        recognizer = (
            cv2.face
            .LBPHFaceRecognizer_create()
        )

        recognizer.train(
            training_images,
            np.array(
                labels,
                dtype=np.int32
            )
        )

        recognizer.save(
            FACE_MODEL_PATH
        )

        print(
            "Face model trained successfully."
        )

        return True

    except Exception as error:

        print(
            "Face model training error:",
            error
        )

        return False

    finally:

        if cursor:

            try:
                cursor.close()

            except Exception:
                pass

        if db:

            try:
                db.close()

            except Exception:
                pass


# ==========================================================
# 9. HOME
# ==========================================================

@app.route("/")
def home():

    if "user_id" in session:

        return redirect(
            url_for("dashboard")
        )

    return redirect(
        url_for("login")
    )


# ==========================================================
# 10. REGISTER / SIGNUP
# ==========================================================

@app.route(
    "/signup",
    methods=["GET", "POST"]
)
@app.route(
    "/register",
    methods=["GET", "POST"]
)
def signup():

    if "user_id" in session:

        return redirect(
            url_for("face_register")
        )

    if request.method == "POST":

        username = request.form.get(
            "username",
            ""
        ).strip()

        email = request.form.get(
            "email",
            ""
        ).strip().lower()

        password = request.form.get(
            "password",
            ""
        )

        if (
            not username
            or not email
            or not password
        ):

            flash(
                "Please fill in all fields.",
                "error"
            )

            return render_template(
                "signup.html"
            )

        if len(password) < 8:

            flash(
                "Password must contain at least 8 characters.",
                "error"
            )

            return render_template(
                "signup.html"
            )

        db = None
        cursor = None

        try:

            db = get_db()

            cursor = db.cursor(
                dictionary=True
            )

            cursor.execute("""
                SELECT id
                FROM users
                WHERE username = %s
                OR email = %s
            """, (
                username,
                email
            ))

            existing_user = cursor.fetchone()

            if existing_user:

                flash(
                    "Username or email is already registered.",
                    "error"
                )

                return render_template(
                    "signup.html"
                )

            hashed_password = (
                generate_password_hash(
                    password
                )
            )

            cursor.execute("""
                INSERT INTO users (
                    username,
                    email,
                    password
                )
                VALUES (%s, %s, %s)
            """, (
                username,
                email,
                hashed_password
            ))

            db.commit()

            new_user_id = (
                cursor.lastrowid
            )

            session.clear()

            session["user_id"] = new_user_id
            session["username"] = username
            session["email"] = email

            session[
                "face_registration_pending"
            ] = True

            flash(
                "Account created. Please register your face.",
                "success"
            )

            return redirect(
                url_for("face_register")
            )

        except Exception as error:

            if db:

                db.rollback()

            print(
                "Signup error:",
                error
            )

            flash(
                "Registration failed. Please try again.",
                "error"
            )

            return render_template(
                "signup.html"
            )

        finally:

            if cursor:

                try:
                    cursor.close()
                except Exception:
                    pass

    return render_template(
        "signup.html"
    )


# ==========================================================
# 11. LOGIN
# ==========================================================

@app.route(
    "/login",
    methods=["GET", "POST"]
)
def login():

    if "user_id" in session:

        return redirect(
            url_for("dashboard")
        )

    if request.method == "POST":

        login_value = (
            request.form.get("email")
            or request.form.get("username")
            or ""
        ).strip()

        password = request.form.get(
            "password",
            ""
        )

        if (
            not login_value
            or not password
        ):

            flash(
                "Enter your username/email and password.",
                "error"
            )

            return render_template(
                "login.html"
            )

        db = get_db()

        cursor = db.cursor(
            dictionary=True
        )

        try:

            cursor.execute("""
                SELECT *
                FROM users
                WHERE email = %s
                OR username = %s
            """, (
                login_value.lower(),
                login_value
            ))

            user = cursor.fetchone()

            if user and check_password_hash(
                user["password"],
                password
            ):

                session.clear()

                session["user_id"] = user["id"]
                session["username"] = user["username"]
                session["email"] = user["email"]

                return redirect(
                    url_for("dashboard")
                )

            flash(
                "Invalid username/email or password.",
                "error"
            )

        finally:

            cursor.close()

    return render_template(
        "login.html"
    )
# ==========================================================
# DASHBOARD
# ==========================================================

@app.route("/dashboard")
@login_required
def dashboard():

    db = None
    cursor = None

    try:
        db = get_db()
        cursor = db.cursor(dictionary=True)

        user_id = session["user_id"]

        # Get logged-in user information
        cursor.execute("""
            SELECT *
            FROM users
            WHERE id = %s
        """, (user_id,))

        user = cursor.fetchone()

        # Count reports
        cursor.execute("""
            SELECT COUNT(*) AS total
            FROM reports
            WHERE user_id = %s
        """, (user_id,))

        report_result = cursor.fetchone()
        total_reports = report_result["total"] if report_result else 0

        # Count drowsiness reports
        cursor.execute("""
            SELECT COUNT(*) AS total
            FROM reports
            WHERE user_id = %s
            AND LOWER(status) LIKE '%drows%'
        """, (user_id,))

        drowsy_result = cursor.fetchone()
        drowsiness_count = (
            drowsy_result["total"]
            if drowsy_result
            else 0
        )

        # Count normal reports
        cursor.execute("""
            SELECT COUNT(*) AS total
            FROM reports
            WHERE user_id = %s
            AND LOWER(status) LIKE '%normal%'
        """, (user_id,))

        normal_result = cursor.fetchone()
        normal_count = (
            normal_result["total"]
            if normal_result
            else 0
        )

        return render_template(
            "dashboard.html",
            user=user,
            username=session.get("username", ""),
            total_reports=total_reports,
            drowsiness_count=drowsiness_count,
            normal_count=normal_count
        )

    except Exception as error:

        print("Dashboard error:", error)

        return render_template(
            "dashboard.html",
            user=None,
            username=session.get("username", ""),
            total_reports=0,
            drowsiness_count=0,
            normal_count=0
        )

    finally:

        if cursor:
            try:
                cursor.close()
            except Exception:
                pass


# ==========================================================
# FACE REGISTRATION PAGE
# ==========================================================

@app.route("/face-register")
@login_required
def face_register():

    db = None
    cursor = None

    try:

        db = get_db()
        cursor = db.cursor()

        user_id = session["user_id"]

        cursor.execute("""
            SELECT COUNT(*)
            FROM face_samples
            WHERE user_id = %s
        """, (user_id,))

        result = cursor.fetchone()

        sample_count = result[0] if result else 0

        return render_template(
            "face_register.html",
            sample_count=sample_count,
            required_samples=REQUIRED_SAMPLES,
            username=session.get("username", "")
        )

    except Exception as error:

        print("Face register page error:", error)

        return render_template(
            "face_register.html",
            sample_count=0,
            required_samples=REQUIRED_SAMPLES,
            username=session.get("username", "")
        )

    finally:

        if cursor:
            try:
                cursor.close()
            except Exception:
                pass


# ==========================================================
# SAVE FACE SAMPLES
# ==========================================================

@app.route("/save-face", methods=["POST"])
@login_required
def save_face():

    db = None
    cursor = None

    try:

        data = request.get_json(silent=True)

        if not data:
            return jsonify({
                "success": False,
                "message": "No face data received."
            }), 400

        images = data.get("images", [])

        if not isinstance(images, list):
            return jsonify({
                "success": False,
                "message": "Invalid image data."
            }), 400

        if len(images) < REQUIRED_SAMPLES:
            return jsonify({
                "success": False,
                "message": (
                    f"Please capture at least "
                    f"{REQUIRED_SAMPLES} face samples."
                )
            }), 400

        user_id = session["user_id"]

        db = get_db()
        cursor = db.cursor()

        # Remove previous samples of this user
        cursor.execute("""
            DELETE FROM face_samples
            WHERE user_id = %s
        """, (user_id,))

        saved_count = 0

        for image_data in images[:REQUIRED_SAMPLES]:

            try:

                # Remove data URL prefix
                if "," in image_data:
                    image_data = image_data.split(",", 1)[1]

                image_bytes = base64.b64decode(image_data)

                image_array = np.frombuffer(
                    image_bytes,
                    dtype=np.uint8
                )

                frame = cv2.imdecode(
                    image_array,
                    cv2.IMREAD_COLOR
                )

                if frame is None:
                    continue

                face_crop = get_face_crop(frame)

                if face_crop is None:
                    continue

                success, encoded_image = cv2.imencode(
                    ".png",
                    face_crop
                )

                if not success:
                    continue

                image_blob = encoded_image.tobytes()

                cursor.execute("""
                    INSERT INTO face_samples (
                        user_id,
                        image
                    )
                    VALUES (%s, %s)
                """, (
                    user_id,
                    image_blob
                ))

                saved_count += 1

            except Exception as image_error:

                print(
                    "Individual face sample error:",
                    image_error
                )

        if saved_count < REQUIRED_SAMPLES:

            db.rollback()

            return jsonify({
                "success": False,
                "message": (
                    f"Only {saved_count} valid face samples "
                    f"were detected. Please capture your face "
                    f"again with good lighting."
                )
            }), 400

        db.commit()

        # Train LBPH model
        training_success = train_face_model()

        if not training_success:

            return jsonify({
                "success": False,
                "message": (
                    "Face samples were saved, but model "
                    "training failed. Check the VS Code terminal."
                )
            }), 500

        # Face registration completed
        session["face_registration_pending"] = False

        return jsonify({
            "success": True,
            "message": (
                f"{saved_count} face samples saved and "
                "face model trained successfully."
            ),
            "sample_count": saved_count
        })

    except Exception as error:

        if db:
            try:
                db.rollback()
            except Exception:
                pass

        print("Save face error:", error)

        return jsonify({
            "success": False,
            "message": "Unable to save face samples."
        }), 500

    finally:

        if cursor:
            try:
                cursor.close()
            except Exception:
                pass


# ==========================================================
# CAMERA PAGE
# ==========================================================

@app.route("/camera")
@login_required
def camera():

    return render_template(
        "camera.html",
        username=session.get("username", "")
    )


# ==========================================================
# LIVE VIDEO STREAM
# ==========================================================
@app.route("/video_feed")
@login_required
def video_feed():

    username = session.get(
        "username",
        "Unknown"
    )

    user_id = session.get(
        "user_id",
        0
    )

    return Response(
        generate_frames(
            username=username,
            user_id=user_id
        ),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )

# ==========================================================
# LIVE CAMERA ALIAS
# ==========================================================

@app.route("/live-camera")
@login_required
def live_camera():

    return render_template(
        "camera.html",
        username=session.get("username", "")
    )


# ==========================================================
# EYE DETECTION PAGE
# ==========================================================

@app.route("/eye-detection")
@login_required
def eye_detection():

    return render_template(
        "eye_detection.html",
        username=session.get("username", "")
    )


# ==========================================================
# DROWSINESS DETECTION PAGE
# ==========================================================

@app.route("/drowsiness")
@login_required
def drowsiness():

    return render_template(
        "drowsiness.html",
        username=session.get("username", "")
    )


# ==========================================================
# START DETECTION
# ==========================================================

@app.route("/start-detection")
@login_required
def start_detection():

    return render_template(
        "drowsiness.html",
        username=session.get("username", "")
    )


# ==========================================================
# CHECK FACE REGISTRATION STATUS
# ==========================================================

@app.route("/face-status")
@login_required
def face_status():

    db = None
    cursor = None

    try:

        db = get_db()
        cursor = db.cursor()

        user_id = session["user_id"]

        cursor.execute("""
            SELECT COUNT(*)
            FROM face_samples
            WHERE user_id = %s
        """, (user_id,))

        result = cursor.fetchone()

        sample_count = result[0] if result else 0

        model_exists = os.path.exists(FACE_MODEL_PATH)

        registered = (
            sample_count >= REQUIRED_SAMPLES
            and model_exists
        )

        return jsonify({
            "success": True,
            "registered": registered,
            "sample_count": sample_count,
            "required_samples": REQUIRED_SAMPLES,
            "model_exists": model_exists
        })

    except Exception as error:

        print("Face status error:", error)

        return jsonify({
            "success": False,
            "registered": False,
            "sample_count": 0,
            "required_samples": REQUIRED_SAMPLES,
            "model_exists": False
        }), 500

    finally:

        if cursor:
            try:
                cursor.close()
            except Exception:
                pass


# ==========================================================
# USER PROFILE
# ==========================================================

@app.route("/profile")
@login_required
def profile():

    db = None
    cursor = None

    try:
        user_id = session["user_id"]

        db = get_db()
        cursor = db.cursor(dictionary=True)

        cursor.execute("""
            SELECT
                id,
                username,
                email,
                full_name,
                phone_number,
                date_of_birth,
                gender,
                licence_id,
                licence_type,
                licence_issue_date,
                licence_expiry_date,
                vehicle_number,
                experience,
                registration_date
            FROM users
            WHERE id = %s
        """, (user_id,))

        user = cursor.fetchone()

        if not user:
            flash("User profile not found.", "error")
            return redirect(url_for("dashboard"))

        return render_template(
            "profile.html",
            user=user,
            profile=user
        )

    except Exception as error:

        print("Profile error:", error)

        flash(
            "Unable to load profile.",
            "error"
        )

        return redirect(
            url_for("dashboard")
        )

    finally:

        if cursor:
            try:
                cursor.close()
            except Exception:
                pass


# ==========================================================
# UPDATE PROFILE
# ==========================================================

@app.route("/update-profile", methods=["POST"])
@login_required
def update_profile():

    db = None
    cursor = None

    try:

        user_id = session["user_id"]

        full_name = request.form.get(
            "full_name", ""
        ).strip()

        phone_number = request.form.get(
            "phone_number", ""
        ).strip()

        date_of_birth = request.form.get(
            "date_of_birth", ""
        ).strip()

        gender = request.form.get(
            "gender", ""
        ).strip()

        licence_id = request.form.get(
            "licence_id", ""
        ).strip()

        licence_type = request.form.get(
            "licence_type", ""
        ).strip()

        licence_issue_date = request.form.get(
            "licence_issue_date", ""
        ).strip()

        licence_expiry_date = request.form.get(
            "licence_expiry_date", ""
        ).strip()

        vehicle_number = request.form.get(
            "vehicle_number", ""
        ).strip()

        experience_value = request.form.get(
            "experience", "0"
        ).strip()

        try:
            experience = int(experience_value)
        except ValueError:
            experience = 0

        db = get_db()
        cursor = db.cursor()

        cursor.execute("""
            UPDATE users
            SET
                full_name = %s,
                phone_number = %s,
                date_of_birth = %s,
                gender = %s,
                licence_id = %s,
                licence_type = %s,
                licence_issue_date = %s,
                licence_expiry_date = %s,
                vehicle_number = %s,
                experience = %s
            WHERE id = %s
        """, (
            full_name,
            phone_number,
            date_of_birth,
            gender,
            licence_id,
            licence_type,
            licence_issue_date,
            licence_expiry_date,
            vehicle_number,
            experience,
            user_id
        ))

        db.commit()

        flash(
            "Profile updated successfully.",
            "success"
        )

        return redirect(url_for("profile"))

    except Exception as error:

        if db:
            try:
                db.rollback()
            except Exception:
                pass

        print("Update profile error:", error)

        flash(
            "Unable to update profile.",
            "error"
        )

        return redirect(url_for("profile"))

    finally:

        if cursor:
            try:
                cursor.close()
            except Exception:
                pass
# ==========================================================
# REPORTS PAGE
# ==========================================================

@app.route("/reports")
@login_required
def reports():

    db = None
    cursor = None

    try:

        user_id = session["user_id"]
        username = session.get("username", "")

        # -------------------------------
        # Pagination
        # -------------------------------

        try:
            page = int(request.args.get("page", 1))
        except (TypeError, ValueError):
            page = 1

        if page < 1:
            page = 1

        page_size = 10
        offset = (page - 1) * page_size

        db = get_db()
        cursor = db.cursor(dictionary=True)

        # -------------------------------
        # Total Reports
        # -------------------------------

        cursor.execute("""
            SELECT COUNT(*) AS total
            FROM reports
            WHERE user_id = %s
        """, (user_id,))

        result = cursor.fetchone()

        total_reports = (
            result["total"]
            if result
            else 0
        )

        # -------------------------------
        # Drowsiness Reports
        # -------------------------------

        cursor.execute("""
            SELECT COUNT(*) AS total
            FROM reports
            WHERE user_id = %s
            AND LOWER(status) LIKE '%drows%'
        """, (user_id,))

        result = cursor.fetchone()

        drowsiness_count = (
            result["total"]
            if result
            else 0
        )

        # -------------------------------
        # Normal Reports
        # -------------------------------

        cursor.execute("""
            SELECT COUNT(*) AS total
            FROM reports
            WHERE user_id = %s
            AND LOWER(status) LIKE '%normal%'
        """, (user_id,))

        result = cursor.fetchone()

        normal_count = (
            result["total"]
            if result
            else 0
        )

        # -------------------------------
        # Calculate Total Pages
        # -------------------------------

        total_pages = (
            (total_reports + page_size - 1)
            // page_size
        )

        if total_pages == 0:
            total_pages = 1

        if page > total_pages:
            page = total_pages
            offset = (page - 1) * page_size

        # -------------------------------
        # Get Reports
        # -------------------------------

        cursor.execute("""
            SELECT
                id,
                user_id,
                username,
                status,
                created_at
            FROM reports
            WHERE user_id = %s
            ORDER BY created_at DESC, id DESC
            LIMIT %s OFFSET %s
        """, (
            user_id,
            page_size,
            offset
        ))

        report_rows = cursor.fetchall()

        # -------------------------------
        # Send Reports To HTML
        # -------------------------------

        return render_template(
            "reports.html",
            reports=report_rows,
            total_reports=total_reports,
            drowsiness_count=drowsiness_count,
            normal_count=normal_count,
            username=username,
            page=page,
            total_pages=total_pages
        )

    except Exception as error:

        print("Reports error:", error)

        return render_template(
            "reports.html",
            reports=[],
            total_reports=0,
            drowsiness_count=0,
            normal_count=0,
            username=session.get("username", ""),
            page=1,
            total_pages=1
        )

    finally:

        if cursor:
            try:
                cursor.close()
            except Exception:
                pass


# ==========================================================
# SAVE REPORT
# ==========================================================

@app.route("/save-report", methods=["POST"])
@login_required
def save_report():

    db = None
    cursor = None

    try:

        user_id = session["user_id"]
        username = session.get("username", "")

        # --------------------------------
        # Get JSON data
        # --------------------------------

        data = request.get_json(silent=True)

        if data:

            status = data.get(
                "status",
                "Normal"
            ).strip()

        else:

            status = request.form.get(
                "status",
                "Normal"
            ).strip()

        if not status:
            status = "Normal"

        db = get_db()
        cursor = db.cursor()

        # --------------------------------
        # Save report
        # --------------------------------

        cursor.execute("""
            INSERT INTO reports (
                user_id,
                username,
                status,
                created_at
            )
            VALUES (
                %s,
                %s,
                %s,
                NOW()
            )
        """, (
            user_id,
            username,
            status
        ))

        db.commit()

        print(
            f"REPORT SAVED | User: {username} | Status: {status}"
        )

        return jsonify({
            "success": True,
            "message": "Report saved successfully."
        })

    except Exception as error:

        if db:
            try:
                db.rollback()
            except Exception:
                pass

        print("Save report error:", error)

        return jsonify({
            "success": False,
            "message": "Unable to save report."
        }), 500

    finally:

        if cursor:
            try:
                cursor.close()
            except Exception:
                pass


# ==========================================================
# CLEAR REPORTS
# ==========================================================

@app.route("/clear-reports", methods=["POST"])
@login_required
def clear_reports():

    db = None
    cursor = None

    try:

        user_id = session["user_id"]

        db = get_db()
        cursor = db.cursor()

        # Delete only logged-in user's reports
        cursor.execute("""
            DELETE FROM reports
            WHERE user_id = %s
        """, (user_id,))

        deleted_count = cursor.rowcount

        db.commit()

        flash(
            f"{deleted_count} report(s) cleared successfully.",
            "success"
        )

        return redirect(
            url_for("reports")
        )

    except Exception as error:

        if db:
            try:
                db.rollback()
            except Exception:
                pass

        print("Clear reports error:", error)

        flash(
            "Unable to clear reports.",
            "error"
        )

        return redirect(
            url_for("reports")
        )

    finally:

        if cursor:
            try:
                cursor.close()
            except Exception:
                pass


# ==========================================================
# SINGLE REPORT DELETE
# ==========================================================

@app.route(
    "/delete-report/<int:report_id>",
    methods=["POST"]
)
@login_required
def delete_report(report_id):

    db = None
    cursor = None

    try:

        user_id = session["user_id"]

        db = get_db()
        cursor = db.cursor()

        cursor.execute("""
            DELETE FROM reports
            WHERE id = %s
            AND user_id = %s
        """, (
            report_id,
            user_id
        ))

        db.commit()

        flash(
            "Report deleted successfully.",
            "success"
        )

        return redirect(
            url_for("reports")
        )

    except Exception as error:

        if db:
            try:
                db.rollback()
            except Exception:
                pass

        print("Delete report error:", error)

        flash(
            "Unable to delete report.",
            "error"
        )

        return redirect(
            url_for("reports")
        )

    finally:

        if cursor:
            try:
                cursor.close()
            except Exception:
                pass


# ==========================================================
# LOGOUT
# ==========================================================

@app.route("/logout")
def logout():

    session.clear()

    flash(
        "You have been logged out successfully.",
        "success"
    )

    return redirect(
        url_for("login")
    )


# ==========================================================
# API - CURRENT USER
# ==========================================================

@app.route("/api/current-user")
@login_required
def current_user():

    return jsonify({
        "success": True,
        "user_id": session.get("user_id"),
        "username": session.get("username"),
        "email": session.get("email")
    })


# ==========================================================
# API - REPORT SUMMARY
# ==========================================================

@app.route("/api/report-summary")
@login_required
def report_summary():

    db = None
    cursor = None

    try:

        user_id = session["user_id"]

        db = get_db()
        cursor = db.cursor(dictionary=True)

        cursor.execute("""
            SELECT
                COUNT(*) AS total_reports,
                SUM(
                    CASE
                        WHEN LOWER(status) LIKE '%drows%'
                        THEN 1
                        ELSE 0
                    END
                ) AS drowsiness_count,
                SUM(
                    CASE
                        WHEN LOWER(status) LIKE '%normal%'
                        THEN 1
                        ELSE 0
                    END
                ) AS normal_count
            FROM reports
            WHERE user_id = %s
        """, (user_id,))

        result = cursor.fetchone()

        return jsonify({
            "success": True,
            "total_reports": result["total_reports"] or 0,
            "drowsiness_count": result["drowsiness_count"] or 0,
            "normal_count": result["normal_count"] or 0
        })

    except Exception as error:

        print(
            "Report summary error:",
            error
        )

        return jsonify({
            "success": False,
            "total_reports": 0,
            "drowsiness_count": 0,
            "normal_count": 0
        }), 500

    finally:

        if cursor:
            try:
                cursor.close()
            except Exception:
                pass


# ==========================================================
# 404 ERROR
# ==========================================================

@app.errorhandler(404)
def page_not_found(error):

    return render_template(
        "404.html"
    ), 404


# ==========================================================
# 500 ERROR
# ==========================================================

@app.errorhandler(500)
def internal_server_error(error):

    return render_template(
        "500.html"
    ), 500


# ==========================================================
# APPLICATION START
# ==========================================================

if __name__ == "__main__":

    print("")
    print("==============================================")
    print(" AI DRIVER DROWSINESS DETECTION SYSTEM")
    print("==============================================")
    print("Database      : MySQL")
    print("Database Name : driver_drowsiness")
    print("Face Model    :", FACE_MODEL_PATH)
    print("==============================================")
    print("Server starting...")
    print("Open: http://127.0.0.1:5000")
    print("==============================================")
    print("")

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True,
        threaded=True
    )