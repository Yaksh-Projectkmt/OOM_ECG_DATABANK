# QA_ECGDATABANK

## Overview

QA_ECGDATABANK is a Django-based web application developed for ECG data
management, analysis, and AI-assisted signal processing. The system
integrates machine learning models, including TensorFlow Lite (TFLite),
to support ECG beat search and ST-segment analysis within a structured
web environment.

This project is designed for research, educational, and clinical data
analysis purposes, providing a scalable framework for handling ECG
datasets and intelligent signal interpretation.

## Key Features

-   Web-based ECG data analysis platform
-   AI/ML model integration using TensorFlow Lite
-   ECG Beat Search functionality
-   ST-Segment analysis module
-   User authentication and session management
-   Structured database management with SQLite
-   Modular Django architecture

## Technology Stack

-   Backend: Python, Django
-   Database: SQLite3
-   Machine Learning: TensorFlow Lite
-   Frontend: HTML, CSS, JavaScript
-   Version Control: Git

## Project Structure

    QA_ECGDATABANK/
    │── manage.py
    │── db.sqlite3
    │
    ├── authuser/           # Authentication and user management
    ├── Beat_Search/        # ECG beat search module
    ├── ST_Segment/         # ST segment analysis module
    ├── Scripts_Models/     # AI models and signal processing scripts
    ├── searchs/            # Main Django project configuration
    └── static/             # Static assets (CSS, JS, Images)

## Installation

### Prerequisites

Ensure the following are installed: - Python 3.8 or higher - pip -
Virtual environment (recommended)

### Step 1: Clone the Repository

    git clone <repository-url>
    cd QA_ECGDATABANK

### Step 2: Create and Activate Virtual Environment

Windows:

    python -m venv venv
    venv\Scripts\activate

Linux / macOS:

    python3 -m venv venv
    source venv/bin/activate

### Step 3: Install Dependencies

If requirements.txt is available:

    pip install -r requirements.txt

Otherwise install manually:

    pip install django tensorflow numpy pillow

## Database Setup

Apply migrations to initialize the database:

    python manage.py migrate

## Environment Configuration (.env Setup)

This project requires environment variables for proper configuration.  
Sensitive settings and local configurations are not included in the cloned repository for security and portability reasons.

### Why .env File is Required
The `.env` file stores important configuration values such as:
- Secret keys
- Debug settings
- Database configurations
- Email settings (if used)
- API keys (if applicable)

These values are intentionally not stored in the repository to protect sensitive data.

## Running the Application

Start the Django development server:

    python manage.py runserver

Access the application in your browser:

    http://192.168.2.96:8081/

## Authentication and Login

The application uses a built-in authentication system for accessing the ECGDATABANK portal.

### Default Login
Existing users can log in using their registered ECGDATABANK username and password through the login page:

### New User Registration
If you are a new user, you must first create an account through the ECGDATABANK portal before accessing the system.

Steps:
1. Open the ECGDATABANK web portal in your browser - https://usfda12c.projectkmt.com/
2. Navigate to the registration/sign-up page
3. Create a new user account with a valid username and password
4. Log in using the newly created credentials

### Admin User Creation (Optional)
If no user exists in the system, create the first user using Django admin:
## AI Models

The project includes pre-trained TensorFlow Lite models for ECG signal
analysis.\
Model files are located in:

    Scripts_Models/Model/

These models support automated ECG pattern recognition and analytical
processing.

## Modules Description

### authuser

Handles user authentication, session management, email utilities, and
API serialization.

### Beat_Search

Provides ECG beat identification and search capabilities using stored
datasets and analytical scripts.

### ST_Segment

Implements ST-segment detection and analysis from ECG signals using
AI-assisted processing.

## Use Cases

-   Medical ECG data research
-   Signal processing experimentation
-   AI model testing for biomedical datasets
-   Academic and clinical data analysis systems

