# ECGDATABANK

ECGDATABANK is a web-based platform for storing, managing, and analyzing
ECG (Electrocardiogram) data. The project is built using Django and
integrates analytical tools and model scripts for ECG data processing,
visualization, and reporting.

## Features

-   ECG data storage and management
-   User authentication and subscription handling
-   ECG analysis and reporting tools
-   Media and static file management
-   Model and script integration for ECG processing
-   Web-based dashboard for data interaction

## Project Structure

    ECGDATABANK/
    ├── analysis_tool/        # ECG analysis modules and tools
    ├── authuser/             # User authentication and management
    ├── ecgdatabank_1/        # Main Django project configuration
    ├── media/                # Uploaded ECG files and media content
    ├── morphology_drow/      # ECG morphology drawing/processing modules
    ├── oom_ecg_data/         # ECG dataset storage
    ├── report/               # Report generation modules
    ├── Scripts_Models/       # ML/DL models and scripts for ECG analysis
    ├── staticfiles/          # Static assets (CSS, JS, images)
    ├── subscription/         # Subscription and plan management
    ├── db.sqlite3            # Default SQLite database
    ├── manage.py             # Django management script
    ├── requirements.txt      # Project dependencies
    └── README.md             # Project documentation

## Requirements

-   Python 3.8 or higher
-   Django (as specified in requirements.txt)
-   pip (Python package manager)

## Installation

1.  Clone the repository:

    ``` bash
    git clone <repository-url>
    cd ECGDATABANK
    ```

2.  Create a virtual environment:

    ``` bash
    python -m venv venv
    ```

3.  Activate the virtual environment:

    -   Windows:

        ``` bash
        venv\Scripts\activate
        ```

    -   Linux/macOS:

        ``` bash
        source venv/bin/activate
        ```

4.  Install dependencies:

    ``` bash
    pip install -r requirements.txt
    ```

## Database Setup

Apply migrations to set up the database:

``` bash
python manage.py migrate
```

Create a superuser (optional, for admin access):

``` bash
python manage.py createsuperuser
```

## Running the Project

Start the development server:

``` bash
python manage.py runserver
```

Open your browser and go to:

    http://127.0.0.1:8000/

## Usage

-   Upload and manage ECG datasets through the web interface
-   Analyze ECG signals using integrated analysis tools
-   Generate reports based on processed ECG data
-   Manage users and subscriptions via the admin panel

## Dependencies

All required Python packages are listed in `requirements.txt`. Install
them using pip before running the project.

## Notes

-   Ensure media and static directories are properly configured in
    Django settings.
-   For production deployment, use a production-ready database and web
    server (e.g., PostgreSQL, Gunicorn, Nginx).

## License

This project is intended for academic and research purposes. Update the
license section according to your distribution needs.
