import logging
import os
import smtplib
from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import List


def create_email(subject: str, recipients: List[str], body: str, body_type: str, sender_email: str = "") -> MIMEMultipart:
    """Creates a basic email structure."""
    message = MIMEMultipart()
    message["From"] = sender_email if sender_email else os.getenv("EMAIL_SENDER", "")
    message["To"] = ", ".join(recipients)
    message["Subject"] = subject
    message.attach(MIMEText(body, body_type))
    return message


def add_attachment(message: MIMEMultipart, path: str):
    """Attaches a file to the email."""
    with open(path, "rb") as attachment:
        part = MIMEBase("application", "octet-stream")
        part.set_payload(attachment.read())
    encoders.encode_base64(part)
    part.add_header("Content-Disposition", f"attachment; filename= {os.path.basename(path)}")
    message.attach(part)


def send_email(message: MIMEMultipart, sender_email: str = "", app_password: str = "", smtp_server="smtp.gmail.com", smtp_port=587):
    """Sends an email using the specified SMTP server."""
    sender_email = sender_email if sender_email else os.getenv("EMAIL_SENDER", "")
    app_password = app_password if app_password else os.getenv("EMAIL_APP_PASSWORD", "")
    try:
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(sender_email, app_password)
            server.sendmail(sender_email, message["To"].split(", "), message.as_string())
        logging.info("Email sent successfully.")
    except Exception as e:
        logging.error(f"Failed to send email: {e}")
