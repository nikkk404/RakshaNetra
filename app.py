# app.py
import secrets
import string
from flask import Flask, request, render_template, jsonify, send_file, url_for, make_response, redirect, flash, send_from_directory
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField
from wtforms.validators import DataRequired, Email, EqualTo, ValidationError
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps 
from fpdf import FPDF 
import io 
import logging
from pymongo import MongoClient
from utils.attack_details import attack_details, load_attack_details
import traceback
import os
import uuid
import click
from werkzeug.utils import secure_filename
from bson.objectid import ObjectId
import torch
from pymongo.errors import PyMongoError
from safetensors.torch import load_file
from transformers import BertTokenizer, BertForSequenceClassification
import pickle
from bson.objectid import ObjectId
from bson.objectid import ObjectId 
# Add near other datetime imports (top of file)
from pymongo import MongoClient
from bson.objectid import ObjectId
from datetime import datetime, timedelta

from geopy.geocoders import Nominatim
import json 
import requests


# Security helpers
import bleach

# Rate limiting
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
#---------------------------------------------------------------------------------------------------------------------------------------------------
models_loaded = False

def load_models_if_needed():
    global models_loaded, tokenizer, category_model, category_encoder, urgency_model, urgency_encoder

    if models_loaded:
        return

    MODELS_DIR = os.path.join(app.root_path, 'models')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load tokenizer
    tokenizer_path = os.path.join(MODELS_DIR, "bert_tokenizer/")
    tokenizer = BertTokenizer.from_pretrained(tokenizer_path)

    # Category model
    category_encoder_path = os.path.join(MODELS_DIR, "label_encoder.pkl")
    with open(category_encoder_path, "rb") as f:
        category_encoder = pickle.load(f)

    category_model_path = os.path.join(MODELS_DIR, "model.safetensors")
    category_model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased", num_labels=len(category_encoder.classes_)
    )
    category_model.load_state_dict(load_file(category_model_path, device="cpu"))
    category_model.to(device)
    category_model.eval()

    # Urgency model
    urgency_encoder_path = os.path.join(MODELS_DIR, "urgency_encoder.pkl")
    with open(urgency_encoder_path, "rb") as f:
        urgency_encoder = pickle.load(f)

    urgency_model_path = os.path.join(MODELS_DIR, "urgency_model.safetensors")
    urgency_model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased", num_labels=len(urgency_encoder.classes_)
    )
    urgency_model.load_state_dict(load_file(urgency_model_path, device="cpu"))
    urgency_model.to(device)
    urgency_model.eval()

    models_loaded = True
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------

app = Flask(__name__)
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize rate limiter (no default global limits; we'll apply per-route)
limiter = Limiter(key_func=get_remote_address, app=app, default_limits=[])


# --- IMPORTANT: Change this to a truly unique, strong secret key for production! ---
app.config['SECRET_KEY'] = 'your_super_secret_secret_key_from_app_py_to_protect_forms_and_sessions_1234567890abcdefgh'
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024   # 10 MB


login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login' 
login_manager.login_message_category = "info" 
login_manager.login_message = "Please log in to access this page." 

# In app.py, replace the OLD UPLOAD_FOLDER_ROOT definition
UPLOAD_FOLDER_ROOT = os.path.join(app.root_path, 'static', 'incident_files') 
UPLOAD_DIR = UPLOAD_FOLDER_ROOT

# Custom Route Definition (Must be present for links to work)
@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    """Serves files from the UPLOAD_DIR (project_root/static/incident_files) directly."""
    try:
        return send_from_directory(UPLOAD_DIR, filename)
    except FileNotFoundError:
        return "File not found.", 404

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'pdf', 'doc', 'docx', 'txt', 'rtf', 'odt', 'csv', 'xlsx'}
# --- MongoDB Atlas Secure Connection ---
from dotenv import load_dotenv
load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")

try:
    client = MongoClient(MONGO_URI)
    db = client["cyber_incident_db"]

    complaints_collection = db['complaints']
    users_collection = db['users']
    feedback_collection = db['feedback']

    logging.info("Connected to MongoDB Atlas successfully.")

except Exception as e:
    logging.error(f"Failed to connect to MongoDB Atlas: {e}")
    exit()

   


class User(UserMixin):
    def __init__(self, id, username, email, password_hash, role='user'):
        self.id = id
        self.username = username
        self.email = email
        self.password_hash = password_hash
        self.role = role

    def get_id(self):
        return str(self.id)

    @property
    def is_admin(self):
        return self.role == 'admin'
    
    @property
    def is_investigator(self):
        return self.role == "investigator"

@login_manager.user_loader
def load_user(user_id):
    try:
        user_data = users_collection.find_one({"_id": ObjectId(user_id)})
        if user_data:
            return User(str(user_data['_id']), user_data['username'], user_data['email'], user_data['password'], user_data.get('role', 'user'))
        return None
    except Exception as e:
        logging.error(f"Error loading user with ID {user_id}: {e}")
        return None

class LoginForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired()])
    password = PasswordField('Password', validators=[DataRequired()])
    submit = SubmitField('Log In')

class RegistrationForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired()])
    email = StringField('Email', validators=[DataRequired(), Email()])
    password = PasswordField('Password', validators=[DataRequired()])
    confirm_password = PasswordField('Confirm Password', validators=[DataRequired(), EqualTo('password')])
    submit = SubmitField('Register')

    def validate_username(self, username):
        user = users_collection.find_one({"username": username.data})
        if user:
            raise ValidationError('That username is taken. Please choose a different one.')

    def validate_email(self, email):
        user = users_collection.find_one({"email": email.data})
        if user:
            raise ValidationError('That email is taken. Please choose a different one.')


# --- Load All Models and Artifacts at Startup ---
try:
    logging.info("Loading all model artifacts...")
    MODELS_DIR = os.path.join(app.root_path, 'models') 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # 1. Load Shared Tokenizer
    tokenizer_path = os.path.join(MODELS_DIR, "bert_tokenizer/")
    tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
    logging.info("Tokenizer loaded successfully.")

    # 2. Load Category Model and its Encoder
    category_model_path = os.path.join(MODELS_DIR, "model.safetensors")
    category_encoder_path = os.path.join(MODELS_DIR, "label_encoder.pkl")
    with open(category_encoder_path, "rb") as f:
        category_encoder = pickle.load(f)
    category_model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=len(category_encoder.classes_))
    category_model.load_state_dict(load_file(category_model_path, device="cpu"))
    category_model.to(device)
    category_model.eval()
    logging.info("Category Model loaded successfully.")

    # 3. Load Urgency Model and its Encoder
    urgency_model_path = os.path.join(MODELS_DIR, "urgency_model.safetensors")
    urgency_encoder_path = os.path.join(MODELS_DIR, "urgency_encoder.pkl")
    with open(urgency_encoder_path, "rb") as f:
        urgency_encoder = pickle.load(f)
    urgency_model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=len(urgency_encoder.classes_))
    urgency_model.load_state_dict(load_file(urgency_model_path, device="cpu"))
    urgency_model.to(device)
    urgency_model.eval()
    logging.info("Urgency Model loaded successfully.")

except Exception as e:
    logging.error(f"FATAL: Could not load ML models. Please ensure you have run the training scripts first. Error: {e}")
    traceback.print_exc()
    exit()

attack_details = load_attack_details()
if not attack_details:
    logging.error("Could not load attack details. Stopping.")
    exit()

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated:
            return redirect(url_for('login', next=request.url))
        if not current_user.is_admin:
            logging.warning(f"User {current_user.username} tried to access admin page without admin role.")
            return "Access Denied: You must be an administrator to view this page.", 403
        return f(*args, **kwargs)
    return decorated_function

def investigator_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated or current_user.role != "investigator":
            return "Access Denied: Investigator only.", 403
        return f(*args, **kwargs)
    return decorated_function

# === Helper for JSON Serialization (Handles ObjectId and datetime for Flask jsonify) ===
def make_json_serializable(data):
    """
    Recursively converts non-JSON-serializable types in MongoDB documents
    (like ObjectId and datetime) to strings, and None to "N/A".
    """
    if isinstance(data, list):
        return [make_json_serializable(item) for item in data]
    elif isinstance(data, dict):
        cleaned_dict = {}
        for key, value in data.items():
            if isinstance(value, ObjectId):
                cleaned_dict[key] = str(value)
            elif isinstance(value, datetime):
                cleaned_dict[key] = value.strftime('%Y-%m-%d %H:%M:%S')
            elif value is None:
                cleaned_dict[key] = "N/A"
            elif isinstance(value, (int, float, str, bool)):
                cleaned_dict[key] = value
            else: # Recursively handle other dicts/lists or special objects
                cleaned_dict[key] = make_json_serializable(value)
        return cleaned_dict
    elif isinstance(data, ObjectId):
        return str(data)
    elif isinstance(data, datetime):
        return data.strftime('%Y-%m-%d %H:%M:%S')
    elif data is None:
        return "N/A"
    return data
# === END make_json_serializable HELPER ===


def generate_secure_user_id(length=8):
    """Generates a secure, random alphanumeric ID."""
    alphabet = string.ascii_letters + string.digits
    return ''.join(secrets.choice(alphabet) for _ in range(length))

# --- START ALL @app.route DEFINITIONS ---

# 1. Basic Public Pages
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/learn') 
def learn(): 
    # This sends all your attack info (info, steps, helplines) to the page
    return render_template('learn.html', attack_topics=attack_details)

# 2. Authentication Routes
@limiter.limit("3 per 10 minutes")
@app.route('/register', methods=['GET', 'POST'])
def register():
    form = RegistrationForm()
    if form.validate_on_submit():

    # --- PASSWORD STRENGTH CHECK (ADD THIS PART) ---
        pw = form.password.data
        if len(pw) < 8:
            form.password.errors.append('Password must be at least 8 characters.')
            return render_template('register.html', title='Register', form=form)
    # OPTIONAL STRONG PASSWORD RULES:
    # import re
    # if not re.search(r"[A-Z]", pw) or not re.search(r"[a-z]", pw) or not re.search(r"[0-9]", pw):
    #     form.password.errors.append('Password must include uppercase, lowercase, and a number.')
    #     return render_template('register.html', title='Register', form=form)

        
        # --- START: Generate Unique User ID ---
        while True:
            # Generate an 8-character alphanumeric ID
            user_id = generate_secure_user_id(8) 
            
            # Check if this ID already exists in the database
            if not users_collection.find_one({"user_id": user_id}):
                # If it doesn't exist, this ID is unique and we can stop looping
                break
        # --- END: Generate Unique User ID ---

        hashed_password = generate_password_hash(form.password.data)
        new_user = {
            "user_id": user_id,  # <-- Here is the new unique ID
            "username": form.username.data,
            "email": form.email.data,
            "password": hashed_password,
            "role": "user" 
        }
        try:
            users_collection.insert_one(new_user)
            # I also added the new user_id to the log message
            logging.info(f"User {form.username.data} (ID: {user_id}) registered successfully!")
            return redirect(url_for('login', message='Registration successful! Please log in.'))
        except PyMongoError as e:
            logging.error(f"MongoDB error during registration: {e}")
            form.username.errors.append('Registration failed due to a database error.')
    return render_template('register.html', title='Register', form=form)


@limiter.limit("2 per 5 minutes")      # IP-based rate limit
@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        if current_user.is_admin:
            return redirect(url_for('admin_overview'))
        elif current_user.is_investigator:
            return redirect(url_for('investigator_overview'))
        else:
            return redirect(url_for('index'))

    form = LoginForm()

    if form.validate_on_submit():

        # --- SANITIZE USERNAME INPUT ---
        username_clean = bleach.clean(form.username.data, strip=True)

        # Pull user record
        user_data = users_collection.find_one({"username": username_clean})

        # Initialize counters if missing
        if user_data:
            failed_count = user_data.get('failed_login_count', 0)
            lockout_until = user_data.get('lockout_until')
        else:
            failed_count = 0
            lockout_until = None

        # --- CHECK ACCOUNT LOCKOUT ---
        if lockout_until:
            try:
                lock_time = datetime.strptime(lockout_until, '%Y-%m-%d %H:%M:%S')
            except Exception:
                lock_time = None

            if lock_time and lock_time > datetime.now():
                flash("Too many failed attempts. Account temporarily locked. Try again later.", "danger")
                logging.warning(f"Locked-out login attempt for {username_clean}")
                return render_template('login.html', title='Login', form=form)

        # --- VERIFY PASSWORD ---
        if user_data and check_password_hash(user_data['password'], form.password.data):

            # SUCCESS → RESET FAILED ATTEMPTS
            users_collection.update_one(
                {"_id": user_data["_id"]},
                {"$set": {"failed_login_count": 0}, "$unset": {"lockout_until": ""}}
            )

            user = User(
                str(user_data['_id']),
                user_data['username'],
                user_data['email'],
                user_data['password'],
                user_data.get('role', 'user')
            )

            login_user(user)
            logging.info(f"User {user.username} logged in. Role: {user.role}")

            next_page = request.args.get('next')

            if user.is_admin:
                return redirect(next_page or url_for('admin_overview'))
            elif user.is_investigator:
                return redirect(next_page or url_for('investigator_overview'))
            else:
                return redirect(next_page or url_for('index'))

        else:
            # --- FAILED LOGIN ATTEMPT ---
            if user_data:
                failed_count = user_data.get('failed_login_count', 0) + 1
                update = {"$set": {"failed_login_count": failed_count}}

                if failed_count >= 5:
                    lockout_time = (datetime.now() + timedelta(minutes=15)).strftime('%Y-%m-%d %H:%M:%S')
                    update["$set"]["lockout_until"] = lockout_time
                    logging.warning(f"User {username_clean} locked out until {lockout_time}")

                users_collection.update_one({"_id": user_data["_id"]}, update)

            logging.warning(f"Login failed for username: {username_clean}")
            flash("Invalid username or password", "danger")

    # Handle success messages (logout etc.)
    message = request.args.get('message')
    if message:
        flash(message, 'success')

    return render_template('login.html', title='Login', form=form)


@app.route('/logout')
@login_required 
def logout():
    logout_user()
    logging.info("User logged out.")
    return redirect(url_for('index'))

# 3. Protected Admin Routes
@app.route('/admin/overview')
@login_required
@admin_required
def admin_overview():
    # --- Status Counts ---
    status_pipeline = [{"$group": {"_id": "$status", "count": {"$sum": 1}}}]
    status_counts = {item['_id']: item['count'] for item in complaints_collection.aggregate(status_pipeline)}
    
    total_cases = sum(status_counts.values())
    pending_cases = status_counts.get("Received", 0) 
    in_progress_cases = status_counts.get("In Progress", 0)
    resolved_cases = status_counts.get("Resolved", 0)

    # --- Urgency Counts ---
    urgency_pipeline = [{"$group": {"_id": "$predicted_urgency", "count": {"$sum": 1}}}]
    urgency_counts = {item['_id']: item['count'] for item in complaints_collection.aggregate(urgency_pipeline)}

    high_urgency_cases = urgency_counts.get("High", 0)
    medium_urgency_cases = urgency_counts.get("Medium", 0)
    low_urgency_cases = urgency_counts.get("Low", 0)
    
    return render_template('admin/overview.html', 
                            active_page='overview', 
                            total_cases=total_cases,
                            pending_cases=pending_cases,
                            in_progress_cases=in_progress_cases,
                            resolved_cases=resolved_cases,
                            high_urgency_cases=high_urgency_cases,
                            medium_urgency_cases=medium_urgency_cases,
                            low_urgency_cases=low_urgency_cases
                        )

@app.route('/admin/manage_cases')
@login_required
@admin_required
def admin_manage_cases():
    # Get filter criteria from URL query parameters
    search_query = request.args.get('search', '').strip()
    selected_status = request.args.get('status', 'all').strip()

    # Build the base query for MongoDB
    mongo_query = {}

    # 1. Add status to the query if a specific status is selected
    if selected_status and selected_status != 'all':
        mongo_query['status'] = selected_status

    # 2. Add search term to the query if provided
    if search_query:
        # Create a regex for case-insensitive search across multiple fields
        regex_query = {"$regex": search_query, "$options": "i"}
        mongo_query['$or'] = [
            {'report_tracking_id': regex_query},
            {'description': regex_query},
            {'predicted_category': regex_query},
            {'name': regex_query}
        ]
    
    # Execute the query and sort by the most recently submitted
    cases = list(complaints_collection.find(mongo_query).sort("date_submitted", -1))
    
    # --- This part remains the same ---
    investigator_map = {}
    investigators = list(users_collection.find({"role": "investigator"}))
    for inv in investigators:
        inv["_id_str"] = str(inv["_id"])
        investigator_map[inv["_id_str"]] = inv["username"]

    for c in cases:
        c["_id_str"] = str(c["_id"])
        assigned_id = c.get("assigned_investigator")
        if assigned_id:
            c["investigator_name"] = investigator_map.get(assigned_id, "N/A")
    
    return render_template(
        "admin/manage-cases.html",
        complaints=cases,
        investigators=investigators,
        # Pass current filter values back to the template
        search_query=search_query,
        selected_status=selected_status
    )

@app.route('/admin/review_reports')
@login_required
@admin_required
def admin_review_reports():
    """
    Display the new page for admins to review reports submitted by investigators.
    """
    # Create a lookup map for investigator names
    investigators = list(users_collection.find({"role": "investigator"}))
    investigator_map = {str(inv['_id']): inv['username'] for inv in investigators}

    # Find all reports pending review
    reports = list(complaints_collection.find(
        {"status": "Pending Admin Review"}
    ).sort("investigator_report_date", 1)) # Sort by oldest first

    # Add the investigator's name to each report object
    for report in reports:
        report['_id_str'] = str(report['_id'])
        investigator_id = report.get('assigned_investigator')
        if investigator_id:
            report['investigator_name'] = investigator_map.get(investigator_id, 'Unknown')
        else:
            report['investigator_name'] = 'N/A'
            
    return render_template('admin/review_reports.html', reports=reports)


@app.route('/api/admin/approve_report/<string:case_id>', methods=['POST'])
@login_required
@admin_required
def api_admin_approve_report(case_id):
    """
    API for an Admin to APPROVE an investigator's final report.
    """
    try:
        case = complaints_collection.find_one({
            "_id": ObjectId(case_id),
            "status": "Pending Admin Review"
        })

        if not case:
            return jsonify({"success": False, "error": "Case not found or not pending review."}), 404

        # This is the NEW file path from the investigator
        investigator_file = case.get('investigator_report_file')
        if not investigator_file:
             return jsonify({"success": False, "error": "No investigator report file found to approve."}), 400

        # --- Approval Logic ---
        result = complaints_collection.update_one(
            {"_id": ObjectId(case_id)},
            {
                "$set": {
                    "status": "Resolved",
                    "final_report_file": investigator_file, # <-- This is the NEW file
                    "final_report_notes": "Report approved by Admin.",
                    "final_report_uploaded_date": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    "final_report_uploaded_by": current_user.username
                },
                "$unset": {
                    "investigator_report_file": "",
                    "investigator_report_date": "",
                    "admin_rejection_reason": "" 
                }
            }
        )
        
        if result.modified_count == 1:
            return jsonify({"success": True})
        else:
            return jsonify({"success": False, "error": "Failed to update case status."})

    except Exception as e:
        logging.error(f"API Error approving report for {case_id}: {e}", exc_info=True)
        return jsonify({"success": False, "error": "A server error occurred."}), 500


@app.route('/api/admin/reject_report/<string:case_id>', methods=['POST'])
@login_required
@admin_required
def api_admin_reject_report(case_id):
    """
    API for an Admin to REJECT an investigator's final report.
    Sets status back to 'In Progress' and logs the rejection reason.
    """
    try:
        data = request.get_json() or {}
        raw_reason = data.get('reason', '')
        reason = bleach.clean(raw_reason, strip=True)
        if not reason:
          return jsonify({"success": False, "error": "A rejection reason is required."}), 400



        case = complaints_collection.find_one({
            "_id": ObjectId(case_id),
            # Check if it's currently pending review to prevent unauthorized changes
            "status": "Pending Admin Review" 
        })

        if not case:
            return jsonify({"success": False, "error": "Case not found or not pending review."}), 404

        # 1. Define the History Entry (Public-Facing Note)
        # Note should be polite but informative
        history_entry = {
            "stage": "In Progress",
            "date": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "notes": "Report rejected by Admin. Investigation reverted to In Progress for revision.",
            "actor": current_user.username
        }

        # 2. --- Rejection Logic ---
        result = complaints_collection.update_one(
            {"_id": ObjectId(case_id)},
            {
                "$set": {
                    # CRITICAL FIX: Set status back to In Progress
                    "status": "In Progress", 
                    "admin_rejection_reason": reason, # Store rejection reason internally
                    "last_updated_by": current_user.username,
                    "last_updated_date": history_entry["date"]
                },
                "$unset": {
                    "investigator_report_file": "",  # Clear the temporary pending file
                    "investigator_report_date": "",
                    "final_report_file": ""          # Ensure final approved file is also clear if it somehow got set
                },
                # Log the rejection to the user's progress tracker
                "$push": {"status_history": history_entry} 
            }
        )
        
        if result.modified_count == 1:
            return jsonify({"success": True, "message": "Report rejected and case moved back to Investigator's queue."})
        else:
            return jsonify({"success": False, "error": "Failed to update case status."})

    except Exception as e:
        logging.error(f"API Error rejecting report for {case_id}: {e}", exc_info=True)
        return jsonify({"success": False, "error": "A server error occurred."}), 500


@app.route('/admin/investigators')
@login_required
@admin_required
def admin_investigators():
    investigators = list(users_collection.find({"role": "investigator"}))
    
    # Define statuses used for calculation
    PENDING_STATUSES = ["Received", "In Progress", "Halted", "Pending Admin Review"]
    SOLVED_STATUSES = ["Resolved"] # Since you removed 'Closed'

    for inv in investigators:
        inv["_id_str"] = str(inv["_id"])
        
        # Find all cases assigned to this investigator
        assigned_cases = list(complaints_collection.find({"assigned_investigator": inv["_id_str"]}))
        
        # Initialize stats
        inv["total_assigned"] = len(assigned_cases)
        inv["total_solved"] = 0
        inv["cases_pending"] = 0 # <-- NEW: Initialize pending count
        total_rating = 0
        rated_cases_count = 0
        
        # Loop through their assigned cases to calculate stats
        for case in assigned_cases:
            current_status = case.get('status')
            
            # 1. Calculate Solved (Final status)
            if current_status in SOLVED_STATUSES:
                inv["total_solved"] += 1
                
                # Check rating (only possible if case is solved)
                if case.get('user_rating'):
                    total_rating += int(case.get('user_rating'))
                    rated_cases_count += 1

            # 2. Calculate Pending (Active workload)
            if current_status in PENDING_STATUSES:
                 inv["cases_pending"] += 1 # <-- NEW: Increment pending count
        
        # Calculate average rating, avoiding division by zero
        if rated_cases_count > 0:
            inv["average_rating"] = round(total_rating / rated_cases_count, 1)
        else:
            inv["average_rating"] = 0
        
        # Pass the number of ratings for context
        inv["rated_cases_count"] = rated_cases_count

    return render_template("admin/investigators.html", investigators=investigators)

@app.route('/admin/add_investigator', methods=['GET', 'POST'])
@login_required
@admin_required
def add_investigator():
    if request.method == 'POST':
        
        user_id_to_promote = bleach.clean(request.form.get('user_id', ''), strip=True)
        if not user_id_to_promote:
            flash('No user selected. Please select a user to promote.', 'warning')
            return redirect(url_for('add_investigator'))
        # optionally validate it's an ObjectId
        try:
            target_obj_id = ObjectId(user_id_to_promote)
        except Exception:
            flash('Invalid user id provided.', 'danger')
            return redirect(url_for('add_investigator'))


        if not user_id_to_promote:
            flash('No user selected. Please select a user to promote.', 'warning')
            return redirect(url_for('add_investigator'))

        # Update the user's role to 'investigator'
        result = users_collection.update_one(
            {'_id': ObjectId(user_id_to_promote), 'role': 'user'},
            {'$set': {'role': 'investigator'}}
        )

        if result.modified_count == 1:
            flash('User successfully promoted to Investigator.', 'success')
        else:
            flash('Could not promote user. They might already be an investigator or admin.', 'danger')
        
        return redirect(url_for('admin_investigators'))

    # For GET request, find all users with the 'user' role
    regular_users = list(users_collection.find({'role': 'user'}))
    for user in regular_users:
        user['_id_str'] = str(user['_id'])
    
    return render_template('admin/add_investigator.html', users=regular_users)



@app.route('/api/admin/assign_case/<string:mongo_id_str>', methods=['POST'])
@login_required
@admin_required
def api_assign_case(mongo_id_str):
        data = request.get_json() or {}
        investigator_id = bleach.clean(str(data.get("investigator_id", "")).strip(), strip=True)
        if not investigator_id:
            return jsonify({"success": False, "error": "No investigator selected."}), 400

        # Validate investigator exists and has role 'investigator'
        inv = users_collection.find_one({"_id": ObjectId(investigator_id)}) if len(investigator_id) == 24 else None
        if not inv or inv.get("role") != "investigator":
            # If your front-end sends a user _id string, ensure it exists; if it sends something else (like user_id), adapt accordingly.
            # Fallback: check by username/user_id fields if you use those instead of ObjectId
            # For now, return an error to prevent assigning invalid IDs
            return jsonify({"success": False, "error": "Investigator not found or invalid ID."}), 400

    
        # 1. Define the History Entry and New Status
        new_status = "In Progress"
        history_entry = {
            "stage": new_status,
            "date": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "notes": "Case assigned to Investigator and investigation has begun.",
            "actor": current_user.username 
        }
        
        # 2. Update MongoDB: Set investigator ID, Status, AND push to history
        result = complaints_collection.update_one(
            {"_id": ObjectId(mongo_id_str)},
            {"$set": 
                {"assigned_investigator": investigator_id,
                "status": new_status # <-- FIX: Set status to In Progress
                },
            "$push": {"status_history": history_entry} # <-- FIX: Log to timeline
            }
        )
        
        return jsonify({"success": result.modified_count == 1})

@app.route('/api/admin/revoke_case/<string:mongo_id_str>', methods=['POST'])
@login_required
@admin_required
def api_revoke_case(mongo_id_str):
    result = complaints_collection.update_one(
        {"_id": ObjectId(mongo_id_str)},
        {"$unset": {"assigned_investigator": ""}}
    )
    return jsonify({"success": result.modified_count == 1})

@app.route("/api/admin/get_case_details/<string:case_id>", methods=["GET"])
@login_required
@admin_required
def api_admin_get_case_details(case_id):
    try:
        case = complaints_collection.find_one({"_id": ObjectId(case_id)})
        if not case:
            return jsonify({"success": False, "error": "Case not found."}), 404
        
        case = make_json_serializable(case)
        return jsonify({"success": True, "case": case})

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

# ADMIN API ENDPOINT: Get Report Details (used by 'View' button modal)
@app.route('/api/admin/get_report_details/<string:mongo_id_str>', methods=['GET'])
@login_required
@admin_required
def api_get_report_details(mongo_id_str):
    """
    Fetches a single report's details for the modal view.
    Ensures MongoDB ObjectId and datetime objects are JSON-serializable.
    """
    try:
        report = complaints_collection.find_one({"_id": ObjectId(mongo_id_str)})
        if report:
            # Use the helper function to ensure all fields are JSON serializable
            serializable_report = make_json_serializable(report)
            return jsonify(serializable_report)
        else:
            return jsonify({"error": "Report not found"}), 404
    except Exception as e:
        logging.error(f"API Error fetching report details for {mongo_id_str}: {e}, traceback: {traceback.format_exc()}")
        return jsonify({"error": "Failed to fetch report details", "details": str(e)}), 500
    
# ...existing code...
# In app.py, around the admin API routes


@app.route('/api/admin/update_case_status/<string:mongo_id_str>', methods=['POST'])
@login_required
@admin_required
def api_update_case_status(mongo_id_str):
    try:
        data = request.get_json() or {}
        new_status = data.get('status')
        # Sanitize public and internal notes
        public_note = bleach.clean(data.get('public_note', ''), strip=True)
        internal_halt_message = bleach.clean(data.get('halt_message', ''), strip=True)
        internal_halt_instructions = bleach.clean(data.get('halt_instructions', ''), strip=True)

        
        if not new_status:
            return jsonify({"error": "Status not provided"}), 400

        valid_statuses = ["Received", "In Progress", "Resolved", "Halted", "Pending Admin Review"]
        
        if new_status not in valid_statuses:
            return jsonify({"error": f"Invalid status: {new_status}"}), 400

        # Prepare the data for the MongoDB $set operation
        set_data = {
            "status": new_status,
            "last_updated_by": current_user.username,
            "last_updated_date": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # --- Determine the public status note ---
        status_note = public_note
        if not status_note:
            if new_status == 'Halted':
                 status_note = "Case placed on hold pending required information."
            elif new_status == 'Resolved':
                 status_note = "Your incident has been successfully resolved."
            else:
                 status_note = f"Status updated to {new_status}."


        # Add the halt fields only when status is Halted
        if new_status == "Halted":
            set_data["halt_message"] = internal_halt_message or "Internal hold placed."
            set_data["halt_instructions"] = internal_halt_instructions or "N/A"
            unset_data = None # Skip unset for Halted status
        else:
            # Clear existing halt fields when moving off 'Halted'
            unset_data = {"halt_message": "", "halt_instructions": ""}

        # --- History Update Logic ---
        history_entry = {
            "stage": new_status,
            "date": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "notes": status_note, 
            "actor": current_user.username 
        }

        update_query = {"$set": set_data, "$push": {"status_history": history_entry}}
        
        # Add $unset only if it was prepared
        if unset_data:
            update_query["$unset"] = unset_data
            
        result = complaints_collection.update_one(
            {"_id": ObjectId(mongo_id_str)},
            update_query
        )

        if result.modified_count == 1:
            logging.info(f"Case {mongo_id_str} status updated to {new_status} by {current_user.username}. Public note: '{status_note}'")
            return jsonify({"success": True, "message": f"Case status updated to {new_status}"})
        else:
            return jsonify({"success": False, "message": "Case not found or status already same"}), 404

    except Exception as e:
        logging.error(f"API Error updating case status for {mongo_id_str}: {e}", exc_info=True)
        return jsonify({"error": "Failed to update case status", "details": str(e)}), 500


# --- ADMIN API ENDPOINT to Upload Final Report ---
@app.route('/api/admin/upload_final_report/<string:mongo_id_str>', methods=['POST'])
@login_required
@admin_required
def api_upload_final_report(mongo_id_str):
    try:
        # Check if the complaint exists
        complaint = complaints_collection.find_one({"_id": ObjectId(mongo_id_str)})
        if not complaint:
            return jsonify({"error": "Complaint not found"}), 404

        # Check for file in request
        if 'final_report_file' not in request.files:
            return jsonify({"error": "No file part in the request"}), 400
        
        file = request.files['final_report_file']
        notes = bleach.clean(request.form.get('notes', ''), strip=True)

        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400

        if file and allowed_file(file.filename):
            report_tracking_id = complaint.get("report_tracking_id", mongo_id_str) # Use existing tracking ID or DB ID
            final_report_dir = os.path.join(UPLOAD_FOLDER_ROOT, report_tracking_id, 'final_reports')
            os.makedirs(final_report_dir, exist_ok=True) # Ensure directory exists
            
            unique_filename = f"final_report_{uuid.uuid4()}_{secure_filename(file.filename)}"
            file_save_path = os.path.join(final_report_dir, unique_filename)
            file.save(file_save_path)
            
            # Path to store in DB (relative to static)
            final_report_url_path = os.path.join('uploads', report_tracking_id, 'final_reports', unique_filename).replace('\\', '/')

            # Update complaint record in MongoDB
            update_data = {
                "final_report_file": final_report_url_path,
                "final_report_notes": notes,
                "final_report_uploaded_date": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                "final_report_uploaded_by": current_user.username
            }
            # Optional: automatically set status to 'Closed' or 'Resolved' after final report upload
            if complaint['status'] not in ["Resolved", "Closed"]:
                update_data["status"] = "Resolved" # Or "Closed" if this is the final closing step
                update_data["last_updated_by"] = current_user.username
                update_data["last_updated_date"] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

            result = complaints_collection.update_one(
                {"_id": ObjectId(mongo_id_str)},
                {"$set": update_data}
            )

            if result.modified_count == 1:
                logging.info(f"Final report for {mongo_id_str} uploaded successfully by {current_user.username}. Path: {final_report_url_path}")
                return jsonify({"success": True, "message": "Final report uploaded successfully!"})
            else:
                return jsonify({"success": False, "message": "Failed to update complaint record after file upload."}), 500
        else:
            return jsonify({"error": "File type not allowed or no file provided"}), 400

    except Exception as e:
        logging.error(f"API Error uploading final report for {mongo_id_str}: {e}, traceback: {traceback.format_exc()}")
        return jsonify({"error": "Failed to upload final report", "details": str(e)}), 500

# --- END ADMIN API ENDPOINT to Upload Final Report ---
@app.route('/api/admin/investigator_pending_cases/<string:investigator_id>', methods=['GET'])
@login_required
@admin_required
def api_investigator_pending_cases(investigator_id):
    """Fetches detailed status of pending cases for a given investigator."""
    
    pending_statuses = ["Received", "In Progress", "Halted"]
    
    # Fetch only the relevant pending cases and important fields
    pending_cases = list(complaints_collection.find({
        "assigned_investigator": investigator_id,
        "status": {"$in": pending_statuses}
    }, {
        "report_tracking_id": 1, 
        "status": 1, 
        "predicted_urgency": 1,
        "description": 1,
        "admin_rejection_reason": 1,
        "halt_message": 1 # Public note on Halt
    }).sort("predicted_urgency", -1))

    # Convert ObjectIds for JSON serialization
    for case in pending_cases:
        case['_id_str'] = str(case['_id'])

    return jsonify(pending_cases)



@app.route('/admin/analytics')
@login_required
@admin_required
def admin_analytics():
    try:
        # --- PREPARE MAP DATA EFFICIENTLY ---
        state_counts = {}
        city_locations = []
        # Fetch all complaints to get location data for mapping
        # Only fetch the fields we absolutely need for this page
        all_complaints = list(complaints_collection.find(
            {}, 
            {
                "state": 1, "location": 1, "latitude": 1, "longitude": 1, 
                "predicted_category": 1, "gender": 1, "age": 1, 
                "status": 1, "predicted_urgency": 1, "date_submitted": 1
            }
        ))
        
        for case in all_complaints:
            # 1. Aggregate State Counts for coloring the map
            state = case.get("state") or "Unknown"
            if state != "Unknown":
                state_counts[state] = state_counts.get(state, 0) + 1

            # 2. NEW: Read pre-saved coordinates directly from the database
            lat = case.get("latitude")
            lng = case.get("longitude")
            
            # Only add a marker if coordinates exist in the document
            if lat is not None and lng is not None:
                city_locations.append({
                    "name": case.get("location", "N/A").title(),
                    "lat": lat,
                    "lng": lng,
                    "attack_type": case.get("predicted_category", "N/A")
                })
        
        # --- The rest of your aggregations remain the same and will work correctly ---
        gender_data = list(complaints_collection.aggregate([
            {"$group": {"_id": "$gender", "count": {"$sum": 1}}}, {"$sort": {"count": -1}}
        ]))
        
        age_data = list(complaints_collection.aggregate([
            {"$project": {"age_num": {"$convert": {"input": "$age", "to": "int", "onError": None, "onNull": None}}}},
            {"$match": {"age_num": {"$ne": None}}},
            {"$bucket": {
                "groupBy": "$age_num", "boundaries": [0, 18, 26, 41, 61, 100],
                "default": "Unknown", "output": {"count": {"$sum": 1}}
            }},
            {"$sort": {"_id": 1}}
        ]))

        attack_type_data = list(complaints_collection.aggregate([
            {"$group": {"_id": "$predicted_category", "count": {"$sum": 1}}}, {"$sort": {"count": -1}}
        ]))

        status_breakdown_data = list(complaints_collection.aggregate([
            {"$group": {"_id": "$status", "count": {"$sum": 1}}}, {"$sort": {"count": -1}}
        ]))

        urgency_breakdown_data = list(complaints_collection.aggregate([
            {"$group": {"_id": "$predicted_urgency", "count": {"$sum": 1}}}, {"$sort": {"count": -1}}
        ]))
        
        monthly_reports_data = list(complaints_collection.aggregate([
            {"$project": {"year_month": {"$substrBytes": ["$date_submitted", 0, 7]}}},
            {"$group": {"_id": "$year_month", "count": {"$sum": 1}}},
            {"$sort": {"_id": 1}}
        ]))

        return render_template(
            'admin/analytics.html',
            active_page='analytics',
            gender_data=gender_data,
            age_data=age_data,
            attack_type_data=attack_type_data,
            state_data_json=json.dumps(state_counts), 
            city_locations_json=json.dumps(city_locations),
            status_breakdown_data=status_breakdown_data, 
            urgency_breakdown_data=urgency_breakdown_data,
            monthly_reports_data=monthly_reports_data 
        )
    except Exception as e:
        logging.error(f"Error generating admin analytics: {e}", traceback.format_exc())
        return render_template('admin/analytics.html', active_page='analytics', error="Failed to load analytics.")

@app.route('/admin/assign_investigator/<string:case_id>', methods=['POST'])
@login_required
@admin_required
def assign_investigator(case_id):
    if not current_user.is_admin:
        return jsonify({"error": "Unauthorized"}), 403

    data = request.get_json()
    investigator_id = data.get("investigator_id")
    if not investigator_id:
        return jsonify({"error": "No investigator selected"}), 400

    db.complaints.update_one(
        {"_id": ObjectId(case_id)},
        {"$set": {"assigned_investigator": investigator_id}}
    )
    return jsonify({"success": True})

@app.route('/admin/revoke_investigator/<string:case_id>', methods=['POST'])
@login_required
def revoke_investigator(case_id):
    if not current_user.is_admin:
        return jsonify({"error": "Unauthorized"}), 403

    db.complaints.update_one(
        {"_id": ObjectId(case_id)},
        {"$unset": {"assigned_investigator": ""}}
    )
    return jsonify({"success": True})

@app.route('/admin/remove_investigator/<string:investigator_id>', methods=['POST'])
@login_required
@admin_required
def remove_investigator(investigator_id):
    # --- START DEBUGGING ---
    investigator_id = bleach.clean(investigator_id, strip=True)

    app.logger.debug(f"--- Attempting to demote user with ID: {investigator_id} ---")
    try:
        user_to_demote = users_collection.find_one({"_id": ObjectId(investigator_id)})
        app.logger.debug(f"Found user in DB: {user_to_demote}")
        
        # Attempt to find the investigator and update their role back to 'user'
        result = users_collection.update_one(
            {"_id": ObjectId(investigator_id), "role": "investigator"},
            {"$set": {"role": "user"}}
        )
        
        app.logger.debug(f"Update result: matched_count={result.matched_count}, modified_count={result.modified_count}")
        # --- END DEBUGGING ---

        if result.modified_count == 1:
            flash('Investigator successfully demoted to a regular user.', 'success')
            return jsonify({"success": True, "message": "Investigator demoted."})
        else:
            return jsonify({"success": False, "error": "User not found or role was not 'investigator'."}), 404

    except Exception as e:
        app.logger.error(f"Error removing investigator role for {investigator_id}: {e}")
        return jsonify({"success": False, "error": "An unexpected server error occurred."}), 500


# NOTE: This assumes `complaints_collection`, `logging`, and `make_json_serializable` are available globally.
from bson.objectid import ObjectId
from functools import cmp_to_key # Needed for custom sorting

from functools import cmp_to_key # Ensure this is imported at the top of app.py

@app.route('/api/admin/investigator_stats/<string:investigator_id>', methods=['GET'])
@login_required
@admin_required
def api_investigator_stats(investigator_id):
    """
    Return assignment / solved / category / urgency / timeline stats for an investigator.
    Includes sorting logic to prioritize pending/active cases in the list.
    """
    try:
        # Define status priority for sorting (lower number = higher priority/more active)
        STATUS_PRIORITY = {
            "Received": 1, 
            "In Progress": 2, 
            "Halted": 3, 
            "Pending Admin Review": 4, 
            "Resolved": 5, 
            "Unknown": 6
        }

        # Define the custom sort function inside the API function's scope:
        def case_priority_sort(a, b):
            # 1. Compare status priority (lower number wins)
            a_priority = STATUS_PRIORITY.get(a.get("status"), 99)
            b_priority = STATUS_PRIORITY.get(b.get("status"), 99)
            
            if a_priority != b_priority:
                return a_priority - b_priority
            
            # 2. If priorities are equal, sort by date (most recent first)
            a_date = a.get("date_submitted", "")
            b_date = b.get("date_submitted", "")
            
            # Simple string comparison for reverse alphabetical order (most recent first)
            if a_date > b_date: return -1
            if a_date < b_date: return 1
            return 0
        # --- End custom sort function definition ---
        
        
        # Fetch cases assigned to this investigator
        cases = list(complaints_collection.find({"assigned_investigator": investigator_id}))
        total_assigned = len(cases)
        solved_count = sum(1 for c in cases if c.get("status") in ["Resolved", "Closed"])
        
        # Aggregations (unchanged logic)
        category_counts = {}
        urgency_counts = {}
        status_counts = {}
        timeline = {} 
        
        for c in cases:
            cat = c.get("predicted_category") or "Uncategorized"
            urgency = c.get("predicted_urgency") or "Unknown"
            status = c.get("status") or "Unknown"
            date = str(c.get("date_submitted", "Unknown")).split(" ")[0]
            
            category_counts[cat] = category_counts.get(cat, 0) + 1
            urgency_counts[urgency] = urgency_counts.get(urgency, 0) + 1
            status_counts[status] = status_counts.get(status, 0) + 1
            timeline[date] = timeline.get(date, 0) + 1

        
        # Include necessary fields, especially Halted notes
        recent_cases_full_data = [{
            "_id": str(c["_id"]),
            "report_tracking_id": c.get("report_tracking_id"),
            "status": c.get("status"),
            "predicted_category": c.get("predicted_category"),
            "predicted_urgency": c.get("predicted_urgency"),
            "date_submitted": c.get("date_submitted"),
            # CRITICAL FIX: Include the halt message and rejection reason for display
            "halt_message": c.get("halt_message"), 
            "admin_rejection_reason": c.get("admin_rejection_reason"),
        } for c in cases]

        # Apply the custom sort to prioritize workload items
        recent_cases = sorted(recent_cases_full_data, key=cmp_to_key(case_priority_sort))[:15] # Limit to 15 items
        
        payload = {
            "investigator_id": investigator_id,
            "total_assigned": total_assigned,
            "solved_count": solved_count,
            "category_counts": category_counts,
            "urgency_counts": urgency_counts,
            "status_counts": status_counts,
            "timeline": timeline,
            "recent_cases": recent_cases # Now correctly sorted and includes necessary notes
        }
        return jsonify(make_json_serializable(payload))
        
    except Exception as e:
        logging.error(f"Error building investigator stats for {investigator_id}: {e}", exc_info=True)
        return jsonify({"error": "Failed to build investigator stats."}), 500


@app.route('/api/admin/investigator_feedback/<string:investigator_id>')
@login_required
@admin_required
def api_investigator_feedback(investigator_id):
    """
    Fetches all user feedback (ratings and comments) for a specific investigator.
    """
    try:
        pipeline = [
            {
                "$match": {
                    "assigned_investigator": investigator_id,
                    "user_rating": {"$exists": True} # Find only cases that have a rating
                }
            },
            {
                "$project": {
                    "_id": 0,
                    "case_id": "$report_tracking_id",
                    "rating": "$user_rating",
                    "comment": "$user_feedback_comment",
                    "date": "$user_rating_date"
                }
            },
            {
                "$sort": {"date": -1} # Show newest feedback first
            }
        ]
        feedback_list = list(complaints_collection.aggregate(pipeline))
        
        # Use the existing helper to handle any datetime objects
        return jsonify(make_json_serializable(feedback_list))
    
    except Exception as e:
        logging.error(f"API Error fetching investigator feedback for {investigator_id}: {e}", exc_info=True)
        return jsonify({"error": "A server error occurred."}), 500

@app.route('/admin/review_feedback')
@login_required
@admin_required
def admin_review_feedback():
    """
    Display all user feedback submitted via the public form, including analytics.
    """
    try:
        # Fetch all feedback documents, newest first
        all_feedback = list(feedback_collection.find().sort("date_submitted", -1))
        
        # Ensure it's ready for JSON/HTML rendering
        for item in all_feedback:
            item['_id_str'] = str(item['_id'])
            # Note: MongoDB dates/ObjectIds are handled by make_json_serializable later if necessary

        # --- NEW: Aggregation Pipeline to calculate averages and total count ---
        feedback_stats = list(feedback_collection.aggregate([
            {
                "$group": {
                    "_id": None, # Group all documents together
                    "total_count": {"$sum": 1},
                    "avg_overall": {"$avg": "$rating_overall"},
                    "avg_satisfaction": {"$avg": "$rating_satisfaction"},
                    "avg_usefulness": {"$avg": "$rating_usefulness"},
                    "avg_ease_of_use": {"$avg": "$rating_ease_of_use"}
                }
            }
        ]))

        # Process statistics for template
        stats = {}
        if feedback_stats:
            stats = feedback_stats[0]
            del stats['_id']
            # Round float values for clean display (e.g., 4.54)
            for key, value in stats.items():
                if isinstance(value, float):
                    stats[key] = round(value, 2)
        
        # --- End Aggregation ---

        return render_template('admin/review_feedback.html', 
                               active_page='feedback', 
                               feedback_list=all_feedback,
                               feedback_stats=stats) # <-- Pass the calculated stats
                               
    except Exception as e:
        # Ensure you handle PyMongoError specifically if needed, but general Exception catches it
        logging.error(f"Error fetching admin feedback: {e}", exc_info=True)
        return render_template('admin/review_feedback.html', 
                               active_page='feedback', 
                               error="Failed to load feedback data.") 
    
@app.route('/investigator/overview')
@login_required
@investigator_required
def investigator_overview():
    inv_id = current_user.get_id()
    cases = list(complaints_collection.find({"assigned_investigator": inv_id}))
    total_cases = len(cases)
    resolved_cases = sum(1 for c in cases if c.get("status") == "Resolved")
    high_urgency = sum(1 for c in cases if c.get("predicted_urgency") == "High")
    medium_urgency = sum(1 for c in cases if c.get("predicted_urgency") == "Medium")
    low_urgency = sum(1 for c in cases if c.get("predicted_urgency") == "Low")
    phishing_cases = sum(1 for c in cases if c.get("predicted_category") == "Phishing")
    return render_template("investigator/overview.html",
                           total_cases=total_cases,
                           resolved_cases=resolved_cases,
                           high_urgency_cases=high_urgency,
                           medium_urgency_cases=medium_urgency,
                           low_urgency_cases=low_urgency,
                           phishing_cases=phishing_cases)

@app.route('/investigator/manage_cases')
@login_required
@investigator_required
def investigator_manage_cases():
    inv_id = current_user.get_id()
    cases = list(complaints_collection.find({"assigned_investigator": inv_id}))

    # --- START: Advanced Sorting Logic ---
    
    # Helper functions to assign a priority score
    def get_urgency_score(case):
        urgency = case.get('predicted_urgency', 'Low')
        if urgency == 'High':
            return 1
        if urgency == 'Medium':
            return 2
        if urgency == 'Low':
            return 3
        return 4 # Other

    def get_status_score(case):
        status = case.get('status', 'Received')
        if status == 'Received':
            return 1
        if status == 'In Progress':
            return 2
        if status == 'Halted':
            return 3
        # --- NEW: Added "Pending Admin Review" to the priority list ---
        if status == 'Pending Admin Review':
            return 4
        if status == 'Resolved':
            return 5
        if status == 'Closed':
            return 6
        return 7 # Other

    # Sort the list of cases using our helper functions
    # This sorts by status first, then by urgency
    sorted_cases = sorted(cases, key=lambda c: (get_status_score(c), get_urgency_score(c)))
    
    # --- END: Advanced Sorting Logic ---

    for c in sorted_cases: # Use the newly sorted list
        c["_id_str"] = str(c["_id"])
        
    return render_template("investigator/manage-cases.html", cases=sorted_cases)

@app.route('/investigator/reports')
@login_required
@investigator_required
def investigator_reports():
    inv_id = current_user.get_id()
    cases = list(complaints_collection.find({"assigned_investigator": inv_id}))
    return render_template("investigator/reports.html", cases=cases)

@app.route('/investigator/analytics')
@login_required
@investigator_required
def investigator_analytics():
    investigator_id = current_user.get_id()

    # Fetch only assigned cases
    cases = list(complaints_collection.find({"assigned_investigator": investigator_id}))

    category_data = {}
    urgency_data = {}
    status_data = {}
    timeline_data = {}

    for case in cases:
        category = case.get("predicted_category", "Uncategorized")
        urgency = case.get("predicted_urgency", "Unknown")
        status = case.get("status", "Unknown")
        # For timeline, we'll just use the date part of the submission timestamp
        date = case.get("date_submitted", "Unknown").split(" ")[0]

        category_data[category] = category_data.get(category, 0) + 1
        urgency_data[urgency] = urgency_data.get(urgency, 0) + 1
        status_data[status] = status_data.get(status, 0) + 1
        timeline_data[date] = timeline_data.get(date, 0) + 1

    return render_template(
        "investigator/analytics.html",
        category_data=category_data,
        urgency_data=urgency_data,
        status_data=status_data,
        timeline_data=timeline_data
    )



# ...existing code...

@app.route('/api/investigator/update_status/<string:case_id>', methods=['POST'])
@login_required
@investigator_required
def api_investigator_update_status(case_id):
    """API endpoint for an investigator to update a case status and log history."""
    try:
        data = request.get_json() or {}
        new_status = data.get('status')
        public_note = bleach.clean(data.get('halt_message', ''), strip=True)
        internal_instructions = bleach.clean(data.get('halt_instructions', ''), strip=True)


        if not new_status:
            return jsonify({"success": False, "error": "New status not provided."}), 400

        # Simplified valid statuses (matching your finalized Admin list)
        valid_statuses = ["Received", "In Progress", "Resolved", "Halted", "Pending Admin Review"]
        
        if new_status not in valid_statuses:
            return jsonify({"success": False, "error": f"Invalid status: {new_status}"}), 400

        # --- Set/Unset Data Preparation ---
        set_data = {
            "status": new_status,
            "last_updated_by": current_user.username,
            "last_updated_date": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        unset_data = {}

        # 1. Handle Halted Status (Requires custom notes)
        if new_status == "Halted":
            # The original validation check should prevent this block from failing if logic is correct
            if not public_note and not internal_instructions:
                return jsonify({"success": False, "error": "Please provide a reason or instruction when setting status to 'Halted'."}), 400
            
            set_data["halt_message"] = public_note or "Case put on hold."
            set_data["halt_instructions"] = internal_instructions or "N/A"
            
            # The status note for the public timeline is the public note
            timeline_note = public_note or "Case put on hold pending investigator action."
            
        # 2. Handle all other statuses (Use generic messages if no public note is sent)
        else:
            # Clear existing halt fields when moving off 'Halted'
            unset_data["halt_message"] = ""
            unset_data["halt_instructions"] = ""
            
            # Use custom note if sent, otherwise use a generic success message
            timeline_note = public_note or f"Status updated to {new_status}." 
        
        
        # --- History Entry and Final Query Build ---
        history_entry = {
            "stage": new_status,
            "date": set_data["last_updated_date"],
            "notes": timeline_note, # The message that appears on the user's progress bar
            "actor": current_user.username 
        }

        update_query = {
            "$set": set_data,
            "$push": {"status_history": history_entry}
        }
        
        if unset_data:
            update_query["$unset"] = unset_data

        # Security: Ensure the case is assigned to the current investigator before updating
        result = complaints_collection.update_one(
            {"_id": ObjectId(case_id), "assigned_investigator": current_user.get_id()},
            update_query
        )

        if result.modified_count == 1:
            return jsonify({"success": True, "message": "Status updated successfully."})
        else:
            return jsonify({"success": False, "error": "Case not found or you are not authorized to update it."}), 404
            
    except Exception as e:
        logging.error(f"Error updating status for case {case_id} by investigator {current_user.id}: {e}", exc_info=True)
        return jsonify({"success": False, "error": "A server error occurred."}), 500


@limiter.limit("10 per hour")
@app.route('/api/investigator/upload_report/<string:case_id>', methods=['POST'])
@login_required
@investigator_required
def api_investigator_upload_report(case_id):
    """API endpoint for an investigator to upload a final report for ADMIN REVIEW, with history logging."""
    try:
        # NOTE: Using UPLOAD_FOLDER_ROOT, complaints_collection, allowed_file, and current_user
        # from your global scope. Assume these are correctly defined globally.

        if 'final_report_file' not in request.files:
            return jsonify({"success": False, "error": "No file part in the request."}), 400
        
        file = request.files['final_report_file']
        if file.filename == '':
            return jsonify({"success": False, "error": "No file selected."}), 400

        # Security Check: 1. Ensure case exists and is assigned to current user.
        case = complaints_collection.find_one(
            {"_id": ObjectId(case_id), "assigned_investigator": current_user.get_id()}
        )
        if not case:
            return jsonify({"success": False, "error": "Case not found or not assigned to you."}), 404

        if file and allowed_file(file.filename):
            report_tracking_id = case.get("report_tracking_id", case_id)
            final_report_dir = os.path.join(UPLOAD_FOLDER_ROOT, report_tracking_id, 'final_reports')
            os.makedirs(final_report_dir, exist_ok=True)
            
            unique_filename = f"final_report_investigator_{uuid.uuid4()}_{secure_filename(file.filename)}"
            file_save_path = os.path.join(final_report_dir, unique_filename) 
            file.save(file_save_path)
            
            final_report_url_path = os.path.join(report_tracking_id, 'final_reports', unique_filename).replace('\\', '/')

            # --- START: WORKFLOW WITH HISTORY LOGGING ---
            
            # 1. Define the NEW STATUS
            new_status = "Pending Admin Review"
            history_entry = {
                "stage": new_status,
                "date": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                "notes": "Investigation complete. Report submitted to Admin for final approval.", 
                "actor": current_user.username 
            }

            # 2. Update MongoDB with $set, $unset, AND $push
            update_result = complaints_collection.update_one(
                {"_id": ObjectId(case_id)},
                {
                    "$set": {
                        "investigator_report_file": final_report_url_path,
                        "investigator_report_date": history_entry["date"], 
                        "status": history_entry["stage"] # <-- CRITICAL: Sets the status to "Pending Admin Review"
                    },
                    "$unset": {
                        "final_report_file": "",
                        "admin_rejection_reason": ""
                    },
                    "$push": {"status_history": history_entry}
                }
            )
            # --- END: WORKFLOW WITH HISTORY LOGGING ---
            
            # 3. Verify database update and handle errors
            if update_result.modified_count == 1:
                return jsonify({"success": True, "message": "Report submitted for admin review."})
            else:
                # DB update failed: Clean up the file saved to disk
                try:
                    os.remove(file_save_path)
                    logging.warning(f"Cleaned up orphaned report file after failed DB update: {file_save_path}")
                except Exception as e:
                    logging.error(f"Failed to delete orphaned file: {e}")
                
                return jsonify({"success": False, "error": "Database update failed after file upload. Report not saved."}), 500
        else:
            return jsonify({"success": False, "error": "File type not allowed."}), 400
            
    except Exception as e:
        logging.error(f"Error uploading report for case {case_id} by investigator {current_user.id}: {e}", exc_info=True)
        return jsonify({"success": False, "error": "A server error occurred."}), 500

@app.route('/api/investigator/get_report_details/<string:mongo_id_str>', methods=['GET'])
@login_required
def api_investigator_get_report_details(mongo_id_str):
    if current_user.role != "investigator":
        return jsonify({"error": "Unauthorized"}), 403
    try:
        report = complaints_collection.find_one({"_id": ObjectId(mongo_id_str)})
        if report:
            return jsonify(make_json_serializable(report))
        else:
            return jsonify({"error": "Report not found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Add this function somewhere before your contact route
def verify_recaptcha(response_key):
    """Verifies the reCAPTCHA response with Google's servers."""
    # This is a placeholder secret key. Replace with your actual secret key.
    RECAPTCHA_SECRET_KEY = '6Lea4tsrAAAAAAX7qXrSB6ZBRIa_RHc4Qfu0I9BT'
    data = {'secret': RECAPTCHA_SECRET_KEY, 'response': response_key}
    try:
        resp = requests.post('https://www.google.com/recaptcha/api/siteverify', data=data, timeout=5)
        resp.raise_for_status()
        result = resp.json()
        return result.get('success', False)
    except requests.exceptions.RequestException as e:
        logging.error(f"reCAPTCHA verification request failed: {e}")
        return False

# This is your new, single contact function
@limiter.limit("5 per minute")     # Prevent brute-force / spam on contact form
@app.route('/contact', methods=['GET', 'POST'])
def contact():
    load_models_if_needed()
    if request.method == 'POST':

        # --- Sanitize all text inputs (XSS protection) ---
        raw_description = request.form.get('description', '')
        description = bleach.clean(raw_description, strip=True)

        name = bleach.clean(request.form.get('name', "Anonymous"), strip=True)
        email = bleach.clean(request.form.get('email', ''), strip=True)
        mobile = bleach.clean(request.form.get('mobile', ''), strip=True)
        gender = bleach.clean(request.form.get('gender', ''), strip=True)
        age = bleach.clean(request.form.get('age', ''), strip=True)
        state = bleach.clean(request.form.get('state', ''), strip=True)
        location_name = bleach.clean(request.form.get('location', ''), strip=True)
        date_of_incident = bleach.clean(request.form.get('date', ''), strip=True)

        # --- 1. reCAPTCHA Verification ---
        captcha_response = request.form.get('g-recaptcha-response')
        if not captcha_response or not verify_recaptcha(captcha_response):
            flash("CAPTCHA verification failed. Please try again.", "danger")
            return render_template('contact.html', captcha_error="CAPTCHA verification failed. Please try again.")

        try:
            # --- 2. Form Processing and Model Prediction ---
            if not description:
                return render_template('contact.html', error="Description is a required field.")

            # (Model prediction logic...)
            inputs = tokenizer(description, return_tensors="pt", truncation=True,
                               padding="max_length", max_length=128)
            inputs = {key: val.to(device) for key, val in inputs.items()}

            with torch.no_grad():
                category_outputs = category_model(**inputs)
                category_idx = torch.argmax(category_outputs.logits, dim=1).item()
                predicted_category = category_encoder.inverse_transform([category_idx])[0]

                urgency_outputs = urgency_model(**inputs)
                urgency_idx = torch.argmax(urgency_outputs.logits, dim=1).item()
                predicted_urgency = urgency_encoder.inverse_transform([urgency_idx])[0]

            # --- 3. Geocoding ---
            latitude, longitude = None, None
            if location_name:
                try:
                    geolocator = Nominatim(user_agent="cyber_crime_portal_contact", timeout=10)
                    location_geo = geolocator.geocode(f"{location_name}, India")

                    if location_geo:
                        latitude = location_geo.latitude
                        longitude = location_geo.longitude

                except Exception as e:
                    logging.warning(f"Could not geocode '{location_name}': {e}")

            # --- 4. Database Document Creation ---
            report_tracking_id = uuid.uuid4().hex[:12].upper()

            complaint = {
                "report_tracking_id": report_tracking_id,
                "status": "Received",
                "name": name,
                "email": email,
                "mobile": mobile,
                "gender": gender,
                "age": age,
                "state": state,
                "location": location_name,
                "latitude": latitude,
                "longitude": longitude,
                "date_of_incident": date_of_incident,
                "description": description,
                "predicted_category": predicted_category,
                "predicted_urgency": predicted_urgency,
                "date_submitted": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                "submitted_by_user_id": current_user.get_id() if current_user.is_authenticated else None,
                "evidence_file": None
            }

            inserted_result = complaints_collection.insert_one(complaint)

            # --- 5. File Upload Handling ---
            if 'evidence_file' in request.files:
                file = request.files['evidence_file']

                if file.filename and allowed_file(file.filename):
                    complaint_upload_dir = os.path.join(UPLOAD_FOLDER_ROOT, report_tracking_id)
                    os.makedirs(complaint_upload_dir, exist_ok=True)

                    unique_filename = f"{uuid.uuid4()}_{secure_filename(file.filename)}"
                    file.save(os.path.join(complaint_upload_dir, unique_filename))

                    evidence_file_path = os.path.join(report_tracking_id, unique_filename).replace('\\', '/')

                    complaints_collection.update_one(
                        {"_id": inserted_result.inserted_id},
                        {"$set": {"evidence_file": evidence_file_path}}
                    )

                    app.logger.debug(
                        f"Evidence file '{unique_filename}' saved and linked to report {report_tracking_id}."
                    )

            success_message = f"Report submitted successfully! Your Tracking ID is: {report_tracking_id}"
            return render_template('contact.html', success=success_message, tracking_id=report_tracking_id)

        except Exception as e:
            logging.error(f"Error saving complaint: {e}", exc_info=True)
            return render_template('contact.html', error="Failed to submit report. Please try again.")

    return render_template('contact.html')



@limiter.limit("5 per minute")     # Prevent spam & brute-force abuse
@app.route('/feedback', methods=['GET', 'POST'])
def feedback():
    if request.method == 'POST':
        try:
            # --- XSS Sanitization for all text inputs ---
            rating = bleach.clean(request.form.get('rating', ''), strip=True)
            satisfaction = bleach.clean(request.form.get('satisfaction', ''), strip=True)
            usefulness = bleach.clean(request.form.get('usefulness', ''), strip=True)
            ease_of_use = bleach.clean(request.form.get('ease_of_use', ''), strip=True)

            suggestions = bleach.clean(request.form.get('suggestions', ''), strip=True)
            contact_email = bleach.clean(request.form.get('contact_email', ''), strip=True)

            # --- Basic validation ---
            if not rating or not satisfaction or not usefulness or not ease_of_use:
                return render_template(
                    'feedback.html',
                    error="Please fill out all required fields (ratings 1–5)."
                )

            feedback_document = {
                "user_id": current_user.get_id() if current_user.is_authenticated else "Anonymous",
                "date_submitted": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                "rating_overall": int(rating),
                "rating_satisfaction": int(satisfaction),
                "rating_usefulness": int(usefulness),
                "rating_ease_of_use": int(ease_of_use),
                "suggestions": suggestions,
                "contact_email": contact_email
            }

            # Insert into DB
            feedback_collection.insert_one(feedback_document)

            return render_template(
                'feedback.html',
                success="Thank you for your valuable feedback!"
            )

        except Exception as e:
            logging.error(f"Error submitting feedback: {e}", exc_info=True)
            return render_template(
                'feedback.html',
                error="Failed to submit feedback due to a server error."
            )

    return render_template('feedback.html')


@app.route('/retrieve_id', methods=['GET', 'POST'])
def retrieve_id():
    reports_found = []
    error_message = None

    if request.method == 'POST':
        # Get all three potential search inputs
        lookup_report_id = request.form.get('report_tracking_id', '').strip().upper()
        lookup_email = bleach.clean(request.form.get('email', '').strip(), strip=True)
        lookup_mobile = bleach.clean(request.form.get('mobile', '').strip(), strip=True)



        query_conditions = {}
        
        # Priority 1: Search ONLY by Report Tracking ID if provided
        if lookup_report_id:
            query_conditions["report_tracking_id"] = lookup_report_id
            
        # Priority 2: If no Report ID, search by Email or Mobile if provided
        elif lookup_email or lookup_mobile:
            contact_or_conditions = []
            if lookup_email:
                contact_or_conditions.append({"email": lookup_email})
            if lookup_mobile:
                contact_or_conditions.append({"mobile": lookup_mobile})
            
            # Apply OR conditions if at least one contact method is given
            if contact_or_conditions:
                query_conditions["$or"] = contact_or_conditions
            else:
                error_message = "Please provide an email, mobile number, or Report ID to search."
                return render_template('retrieve_id.html', reports_found=reports_found, error=error_message)
        
        # Priority 3: If nothing was entered in any field
        else:
            error_message = "Please provide an email, mobile number, or Report ID to search."
            return render_template('retrieve_id.html', reports_found=reports_found, error=error_message)

        # Execute the query if query_conditions were successfully built
        if query_conditions:
            queried_reports = list(complaints_collection.find(query_conditions).sort("date_submitted", -1))
            
            # Deduplicate results if necessary and ensure valid tracking IDs exist
            unique_report_ids_set = set()
            deduplicated_reports = []
            for report in queried_reports:
                report_id = report.get('report_tracking_id') 
                if report_id and report_id not in unique_report_ids_set: 
                    deduplicated_reports.append(report)
                    unique_report_ids_set.add(report_id)
            reports_found = deduplicated_reports

            # Refined error message if nothing was found AFTER query execution
            if not reports_found:
                if lookup_report_id: # Specific message for ID search
                    error_message = f"Report with ID '{lookup_report_id}' not found."
                elif lookup_email or lookup_mobile: # Specific message for contact search
                    error_message = "No reports found with the provided email or mobile number."
                else: # Fallback (less likely to hit with revised flow)
                    error_message = "No search criteria yielded results." 
        else: 
            error_message = "Invalid search criteria provided."

    return render_template('retrieve_id.html', reports_found=reports_found, error=error_message)

# --- NEW ROUTE: My Reports (for logged-in users) ---
@app.route('/my_reports')
@login_required # This page requires user to be logged in
def my_reports():
    try:
        user_id = current_user.get_id()
        reports = list(complaints_collection.find({"submitted_by_user_id": user_id}).sort("date_submitted", -1))
        
        # Ensure ObjectIds are strings for rendering in templates
        for report in reports:
            report['_id_str'] = str(report['_id'])
        
        if not reports:
            message = "You have not submitted any reports yet."
            return render_template('my_reports.html', active_page='my_reports', reports=[], message=message) 
        
        return render_template('my_reports.html', active_page='my_reports', reports=reports)
    except Exception as e:
        logging.error(f"Error fetching reports for user {current_user.username}: {e}", exc_info=True)
        return render_template('my_reports.html', active_page='my_reports', error="Failed to load your reports.")
# --- END NEW ROUTE: My Reports ---

@app.route('/my_profile')
@login_required
def my_profile():
    try:
        # Get the current user's MongoDB _id
        user_mongo_id = ObjectId(current_user.get_id())
        
        # Fetch the FULL user document from the database
        # This will contain the username, email, and new user_id
        user_data = users_collection.find_one({"_id": user_mongo_id})
        
        if user_data:
            # Pass the user_data to the new template
            return render_template('my_profile.html', user_data=user_data, active_page='my_profile')
        else:
            flash("User not found.", "danger")
            return redirect(url_for('index'))
    
    except Exception as e:
        logging.error(f"Error fetching profile for user {current_user.username}: {e}", exc_info=True)
        flash("An error occurred while fetching your profile.", "danger")
        return redirect(url_for('index'))


@app.route('/api/submit_rating/<string:mongo_id_str>', methods=['POST'])
@login_required
def api_submit_rating(mongo_id_str):
    """
    API endpoint for a user to submit a rating for their own resolved case.
    """
    try:
        data = request.get_json() or {}
        rating = data.get('rating')
        comment = bleach.clean(data.get('comment', ''), strip=True)


        if not rating or not (1 <= int(rating) <= 5):
            return jsonify({"success": False, "error": "Invalid rating. Must be between 1 and 5."}), 400

        # --- CRITICAL SECURITY CHECK ---
        # 1. Find the case by its ID
        # 2. Ensure it belongs to the currently logged-in user
        case = complaints_collection.find_one({
            "_id": ObjectId(mongo_id_str),
            "submitted_by_user_id": current_user.get_id() 
        })

        if not case:
            return jsonify({"success": False, "error": "Report not found or you are not authorized to rate it."}), 404

        # 3. Check if the case is actually resolved
        if case.get('status') not in ["Resolved", "Closed"]:
            return jsonify({"success": False, "error": "You can only rate cases that are Resolved or Closed."}), 400
        
        # 4. Check if it has already been rated
        if case.get('user_rating'):
             return jsonify({"success": False, "error": "This case has already been rated."}), 400

        # --- All checks passed. Update the document. ---
        result = complaints_collection.update_one(
            {"_id": ObjectId(mongo_id_str)},
            {"$set": {
                "user_rating": int(rating),
                "user_feedback_comment": comment,
                "user_rating_date": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }}
        )

        if result.modified_count == 1:
            logging.info(f"User {current_user.username} submitted rating ({rating} stars) for case {mongo_id_str}")
            return jsonify({"success": True, "message": "Thank you for your feedback!"})
        else:
            return jsonify({"success": False, "error": "Failed to save rating."}), 500

    except Exception as e:
        logging.error(f"API Error submitting rating for {mongo_id_str} by user {current_user.id}: {e}", exc_info=True)
        return jsonify({"success": False, "error": "A server error occurred."}), 500

@app.route('/predict', methods=['POST'])
def predict():
    logging.info('predict endpoint called')
    description = request.form.get('description', '').strip()
    
    if not description:
        # ... (error handling code) ...
        return render_template('predict.html', error="Please provide a description to predict.")
        
    try: # <--- This opens the try block
        # --- 1. Tokenization and Device Transfer ---
        inputs = tokenizer(description, return_tensors="pt", truncation=True, padding="max_length", max_length=128)
        inputs = {key: val.to(device) for key, val in inputs.items()}
        
        with torch.no_grad():
            # --- 2. Category Model Prediction ---
            category_outputs = category_model(**inputs)
            category_idx = torch.argmax(category_outputs.logits, dim=1).item()
            predicted_category = category_encoder.inverse_transform([category_idx])[0]

            # --- 3. Urgency Model Prediction ---
            urgency_outputs = urgency_model(**inputs)
            urgency_idx = torch.argmax(urgency_outputs.logits, dim=1).item()
            predicted_urgency = urgency_encoder.inverse_transform([urgency_idx])[0]
            
        logging.info(f"Model predictions: Category={predicted_category}, Urgency={predicted_urgency}")
        
        # --- 4. Fetch Attack Details ---
        normalized_prediction = predicted_category.strip().lower()
        normalized_attack_details = {key.strip().lower(): value for key, value in attack_details.items()}
        details = normalized_attack_details.get(normalized_prediction, {
            "info": ["Information not available."],
            "post_attack_steps": ["N/A"],
            "helplines": {"N/A": "N/A"}
        })
        
        # --- 5. Render Template with BOTH Predictions ---
        # NOTE: Your error line (1547) is likely just before this block because the previous line was the last in the 'try'
        return render_template('predict.html',
                               predicted_urgency=predicted_urgency,
                               prediction=predicted_category,
                               info=details['info'],
                               steps=details['post_attack_steps'],
                               helplines=details['helplines'])
                               
    except Exception as e: # <--- This is the missing block!
        logging.error(f"Model prediction error in /predict route: {e}", exc_info=True)
        return render_template('predict.html', error="Prediction Error. Please check server logs.")

import os
import io
from flask import send_file, url_for # Ensure these are imported
from fpdf import FPDF
from bson import ObjectId # Ensure this is imported

import os
import io
from flask import send_file, url_for
from fpdf import FPDF
from bson import ObjectId 

@app.route('/download_receipt/<report_tracking_id>')
def download_receipt(report_tracking_id):
    try:
        from bson import ObjectId
        query_options = [
            {"report_tracking_id": report_tracking_id},
            {"case_id": report_tracking_id}
        ]

        if len(report_tracking_id) == 24:
            try:
                query_options.append({"_id": ObjectId(report_tracking_id)})
            except Exception:
                pass

        report = complaints_collection.find_one({"$or": query_options})

        if not report:
            logging.error(f"Receipt generation failed: Report with ID {report_tracking_id} not found.")
            return f"❌ Report with ID {report_tracking_id} not found in the database.", 404

        logging.info(f"Generating receipt for report ID: {report_tracking_id}")

        # Initialize FPDF
        pdf = FPDF(unit='mm', format='A4')
        pdf.add_page()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.set_font('helvetica', '', 12)

        # Header - Light Blue Background
        pdf.set_fill_color(173, 216, 230)
        pdf.set_text_color(0, 0, 0)
        pdf.set_font('helvetica', 'B', 16)
        pdf.cell(0, 10, 'Incident Report Receipt', align='C', border=0, new_x='LMARGIN', new_y='NEXT', fill=True)
        pdf.ln(5)

        # Tracking ID
        pdf.set_font('helvetica', 'B', 14)
        pdf.cell(0, 8, f"Report Tracking ID: {report.get('report_tracking_id', report.get('case_id', report_tracking_id))}",
                 align='C', border=0, new_x='LMARGIN', new_y='NEXT')
        pdf.ln(8)

        # --- Helper functions (Restored Shading Logic) ---
        def add_field(label, value):
            display_value = str(value) if value not in [None, ""] else "N/A"
            
            # Use light gray background for field rows (Original shading logic)
            fill = pdf.get_y() % (6 * 2) < 6 
            pdf.set_fill_color(240, 240, 240)
            
            pdf.set_font('helvetica', 'B', 10)
            pdf.cell(50, 6, f'{label}:', 0, 0, 'L', fill)
            pdf.set_font('helvetica', '', 10)
            pdf.cell(0, 6, display_value, 0, 1, 'L', fill)

        def add_multiline(label, value):
            display_value = str(value) if value not in [None, ""] else "N/A"
            pdf.set_font('helvetica', 'B', 10)
            pdf.write(6, f'{label}:\n')
            pdf.set_font('helvetica', '', 10)
            pdf.multi_cell(0, 5, display_value)
            pdf.ln(2)

        def add_section_header(title):
            pdf.ln(2)
            pdf.set_fill_color(220, 220, 220) # Slightly darker gray header
            pdf.set_font('helvetica', 'B', 12)
            pdf.cell(0, 7, title, 0, 1, 'L', True)
            pdf.set_font('helvetica', '', 10)
            pdf.ln(2)


        # --- 1. Case Summary ---
        add_section_header('Case Summary')
        add_field('Status', report.get('status', 'N/A'))
        add_field('Predicted Category', report.get('predicted_category', 'N/A'))
        add_field('Predicted Urgency', report.get('predicted_urgency', 'N/A'))
        add_field('Date Submitted', report.get('date_submitted', 'N/A'))
        

        # --- 2. Reporter Details ---
        add_section_header('Reporter Details')
        add_field('Report Type', report.get('registration_type', 'Registered' if report.get('submitted_by_user_id') else 'Anonymous'))
        add_field('Name', report.get('name', 'N/A'))
        add_field('Email', report.get('email', 'N/A'))
        add_field('Mobile', report.get('mobile', 'N/A'))
        add_field('Gender', report.get('gender', 'N/A'))
        add_field('Age', report.get('age', 'N/A'))
        

        # --- 3. Incident Details ---
        add_section_header('Incident Details')
        add_field('State', report.get('state', 'N/A'))
        add_field('Location', report.get('location', 'N/A'))
        add_field('Date of Incident', report.get('date_of_incident', 'N/A'))
        # sanitize text for PDF rendering
        safe_description = bleach.clean(str(report.get('description', 'N/A')), strip=True)
        # collapse long whitespace/newlines to reasonable formatting
        safe_description = ' '.join(safe_description.split())
        add_multiline('Description', safe_description)

        

        # --- 4. Evidence Status (Link Fix Applied) ---
        add_section_header('Evidence Status')
        evidence_file_path = report.get('evidence_file')
        
        if evidence_file_path and evidence_file_path != "N/A":
            evidence_url = url_for('uploaded_file', filename=evidence_file_path, _external=True)
            file_name_display = os.path.basename(evidence_file_path)
            
            pdf.set_font('helvetica', 'B', 10)
            pdf.write(6, f"Evidence Uploaded: {file_name_display}\n")
            
            # Write URL Label
            pdf.write(6, "File URL: ")
            
            # Write URL as a clickable link (smaller font to fit)
            pdf.set_text_color(0, 0, 255) 
            pdf.set_font('helvetica', 'U', 8) 
            pdf.write(6, evidence_url, evidence_url)
            
            # Reset formatting for subsequent text
            pdf.set_text_color(0, 0, 0)
            pdf.set_font('helvetica', '', 10)
            pdf.ln(6) 
        else:
            pdf.write(6, "No evidence file uploaded.\n")


        # --- 5. Final Report Status (Link Fix Applied) ---
        add_section_header('Final Report Status')
        final_report_file = report.get('final_report_file')
        if final_report_file and final_report_file != "N/A":
            final_report_url = url_for('uploaded_file', filename=final_report_file, _external=True)
            final_file_name_display = os.path.basename(final_report_file)
            
            pdf.set_font('helvetica', 'B', 10)
            pdf.write(6, f"Final Report Uploaded: {final_file_name_display}\n")
            pdf.write(6, f"Uploaded On: {report.get('final_report_uploaded_date', 'N/A')}\n")
            pdf.write(6, f"Uploaded By: {report.get('final_report_uploaded_by', 'N/A')}\n")
            
            # Write URL as a clickable link
            pdf.write(6, "File URL: ")
            pdf.set_text_color(0, 0, 255)
            pdf.set_font('helvetica', 'U', 8)
            pdf.write(6, final_report_url, final_report_url)
            
            # Reset formatting
            pdf.set_text_color(0, 0, 0)
            pdf.set_font('helvetica', '', 10)
            pdf.ln(6) 
        else:
            pdf.write(6, "No final report uploaded yet.\n")

        # Output PDF 
        pdf_bytes = pdf.output(dest='S')
        pdf_buffer = io.BytesIO(pdf_bytes)
        pdf_buffer.seek(0)

        return send_file(
            pdf_buffer,
            as_attachment=True,
            download_name=f"Report_Receipt_{report_tracking_id}.pdf",
            mimetype="application/pdf"
        )

    except Exception as e:
        logging.error(f"Error generating PDF for {report_tracking_id}: {e}", exc_info=True)
        return f"⚠️ An error occurred while generating your receipt: {e}", 500


@app.route("/solved-cases-analytics")
def solved_cases_analytics():
    try:
        solved_cases = list(complaints_collection.find({"status": {"$in": ["Resolved", "Closed"]}}))

        state_counts = {}
        city_locations = []

        for case in solved_cases:
            # Aggregate State Counts
            state = case.get("state") or "Unknown"
            if state != "Unknown":
                state_counts[state] = state_counts.get(state, 0) + 1

            # --- NEW: Read coordinates directly from the database ---
            lat = case.get("latitude")
            lng = case.get("longitude")
            
            # Only add a marker if we have valid coordinates
            if lat is not None and lng is not None:
                city_locations.append({
                    "name": case.get("location", "N/A").title(),
                    "lat": lat,
                    "lng": lng,
                    "attack_type": case.get("predicted_category", "N/A")
                })
        
        # --- The rest of the function remains the same ---
        gender_data, age_data, attack_type_data = {}, {}, {}
        for case in solved_cases:
            gender = case.get("gender") or "Unknown"
            gender_data[gender] = gender_data.get(gender, 0) + 1
            attack = case.get("predicted_category") or "Unknown"
            attack_type_data[attack] = attack_type_data.get(attack, 0) + 1
            age = case.get("age")
            group = "Unknown"
            if age:
                try:
                    age_num = int(age)
                    if age_num < 18: group = "Under 18"
                    elif age_num <= 29: group = "18-29"
                    elif age_num <= 44: group = "30-44"
                    elif age_num <= 59: group = "45-59"
                    else: group = "60+"
                except (ValueError, TypeError): pass
            age_data[group] = age_data.get(group, 0) + 1

        return render_template(
            "solved_analytics.html",
            gender_data=gender_data,
            age_data=age_data,
            attack_type_data=attack_type_data,
            state_data_json=json.dumps(state_counts),
            city_locations_json=json.dumps(city_locations)
        )
    except Exception as e:
        logging.error(f"Error in /solved-cases-analytics: {e}", exc_info=True)
        return render_template('error.html', error="Could not load analytics data."), 500

@app.errorhandler(500)
def internal_error(error):
    logging.error(f"Internal server error: {error}, traceback: {traceback.format_exc()}")
    return render_template('error.html', error="An unexpected error occurred."), 500

if __name__ == '__main__':
    app.run(debug=False)


@app.cli.command("create-user")
@click.argument("username")
@click.argument("email")
@click.argument("password")
@click.option("--role", default="user", help="Role for the user (user or admin)")
@click.option("--update", is_flag=True, help="Update existing user's password and role.") # Add this line
def create_user_command(username, email, password, role, update): # Add update here
    """Creates a new user or updates an existing one."""
    try:
        from dotenv import load_dotenv
        load_dotenv()

        MONGO_URI = os.getenv("MONGO_URI")

        cli_client = MongoClient(MONGO_URI)
        cli_db = cli_client['cyber_incident_db']
        cli_users_collection = cli_db['users']

        user = cli_users_collection.find_one({"username": username})
        hashed_password = generate_password_hash(password)

        if user and update:
            # --- UPDATE LOGIC ---
            cli_users_collection.update_one(
                {"username": username},
                {"$set": {"password": hashed_password, "role": role, "email": email}}
            )
            click.echo(f"User '{username}' updated successfully!")
        elif user:
            # --- USER EXISTS, NO UPDATE FLAG ---
            click.echo(f"Error: User '{username}' already exists. Use --update to change password/role.")
        else:
            # --- CREATE NEW USER ---
            if role not in ['user', 'admin', 'investigator']:
                click.echo("Error: Invalid role. Must be 'user', 'admin', or 'investigator'.")
            else:
                new_user = {
                    "username": username, "email": email,
                    "password": hashed_password, "role": role
                }
                cli_users_collection.insert_one(new_user)
                click.echo(f"User '{username}' with role '{role}' created successfully!")
        
        cli_client.close()
    except Exception as e:
        click.echo(f"An error occurred: {e}")

       

