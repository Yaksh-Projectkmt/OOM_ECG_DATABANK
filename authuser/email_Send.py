import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Email configuration
EMAIL_ADDRESS = "aikmtnew@gmail.com"
EMAIL_PASSWORD = "ryvd emev ocbe vwjt"
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587

# -----------------------------------------------------------
# MASTER FUNCTION — Send Email (Auto Detect HTML / Plain Text)
# -----------------------------------------------------------
def send_email(receiver_email, subject, message):
    try:
        msg = MIMEMultipart("alternative")
        msg["From"] = EMAIL_ADDRESS
        msg["To"] = receiver_email
        msg["Subject"] = subject

        # Auto-detect HTML content
        is_html = any(tag in message.lower() for tag in ["<html", "<div", "<p", "<table", "<br"])

        if is_html:
            msg.attach(MIMEText(message, "html"))
        else:
            msg.attach(MIMEText(message, "plain"))

        # Send email securely
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            server.send_message(msg)

    except Exception as e:
        print(f"Failed to send email: {e}")



# -------------------------------------------------------------------------
# COMMON HEADER TEMPLATE — USES LOCAL STATIC IMAGE (Django static reference)
# -------------------------------------------------------------------------
def email_header(title):
    return f"""
    <div style="background:#f5f7fa;padding:20px;font-family:Arial;">
      <div style="max-width:600px;margin:auto;background:#ffffff;padding:25px;border-radius:12px;
      box-shadow:0 4px 12px rgba(0,0,0,0.08);">

        <div style="text-align:center;">
          <h2 style="color:#2a9d8f;margin-top:10px;">{title}</h2>
        </div>
    """


def email_footer():
    return """
        <hr style="border:0;border-top:1px solid #e8e8e8;margin:25px 0;">
        <p style="text-align:center;color:#888;font-size:13px;">
          Support: support@ProjectKmt.com<br>
          © 2025 OOMECG DataBank
        </p>
      </div>
    </div>
    """



# ------------------------------------------------------
# 1. WELCOME EMAIL (After Registration)
# ------------------------------------------------------
def send_welcome_email(username, receiver_email):

    subject = "Welcome to OOMECG DataBank!"

    html = (
        email_header("Welcome to OOMECG DataBank!") +
        f"""
        <p style="font-size:15px;color:#333;">
          Hi <b>{username}</b>,<br><br>
          Thank you for joining <b>OOMECG DataBank</b>.<br>
          Your account has been created successfully.
        </p>

        <p style="font-size:15px;text-align:center;margin-top:25px;color:#444;">
          We will notify you once your documents are verified.
        </p>
        """ +
        email_footer()
    )

    send_email(receiver_email, subject, html)



# ------------------------------------------------------
# 2. DOCUMENT UNDER VERIFICATION EMAIL
# ------------------------------------------------------
def send_documents_processing_email(username, receiver_email):

    subject = "Your Documents Are Under Verification"

    html = (
        email_header("Your Documents Are Being Verified") +
        f"""
        <p style="font-size:15px;color:#333;">
          Hello <b>{username}</b>,<br><br>
          We have received your documents and our verification team is reviewing them.
        </p>

        <p style="font-size:15px;text-align:center;margin-top:20px;color:#555;">
          You will receive another email once validation is complete.
        </p>
        """ +
        email_footer()
    )

    send_email(receiver_email, subject, html)



# ------------------------------------------------------
# 3. REGISTRATION APPROVED EMAIL
# ------------------------------------------------------
def send_approved_email(username, receiver_email):
    subject = "OOMECG DataBank – Registration Approved"

    html = (
        email_header("Registration Approved!") +
        f"""
        <p style="font-size:15px;color:#333;">
            Hi <b>{username}</b>,<br><br>
            Congratulations! Your registration has been <b>approved</b>.<br>
            You now have full access to the OOMECG DataBank platform.
        </p>

        <div style="text-align:center;margin-top:20px;">
            <a href="http://127.0.0.1:8000/auth/login/"
               style="background:#2a9d8f;color:white;padding:12px 25px;text-decoration:none;
               border-radius:6px;font-size:15px;display:inline-block;">
               Login to Website
            </a>
        </div>
        """ +
        email_footer()
    )

    send_email(receiver_email, subject, html)



# ------------------------------------------------------
# 4. REJECTED EMAIL (With Comment)
# ------------------------------------------------------
def send_rejected_email(username, receiver_email, comment=""):
    subject = "OOMECG DataBank – Registration Rejected"

    html = (
        email_header("Registration Rejected") +
        f"""
        <p style="font-size:15px;color:#333;">
            Hi <b>{username}</b>,<br><br>
            Unfortunately, your registration has been rejected.<br><br>
            <b>Reason:</b> {comment}
        </p>
        """ +
        email_footer()
    )

    send_email(receiver_email, subject, html)
