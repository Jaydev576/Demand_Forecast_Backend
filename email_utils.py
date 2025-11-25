from fastapi_mail import FastMail, MessageSchema, ConnectionConfig, MessageType
from settings import settings
from pydantic import EmailStr
from typing import List, Dict, Any
import os

conf = ConnectionConfig(
    MAIL_USERNAME=settings.MAIL_USERNAME,
    MAIL_PASSWORD=settings.MAIL_PASSWORD,
    MAIL_FROM=settings.MAIL_FROM,
    MAIL_PORT=settings.MAIL_PORT,
    MAIL_SERVER=settings.MAIL_SERVER,
    MAIL_STARTTLS=settings.MAIL_STARTTLS,
    MAIL_SSL_TLS=settings.MAIL_SSL_TLS,
    USE_CREDENTIALS=True,
    VALIDATE_CERTS=True,
    TEMPLATE_FOLDER=os.path.join(os.path.dirname(__file__), "templates")
)

async def send_email(recipients: List[EmailStr], subject: str, template_name: str, template_body: Dict[str, Any]):
    message = MessageSchema(
        subject=subject,
        recipients=recipients,
        template_body=template_body,
        subtype=MessageType.html
    )

    fm = FastMail(conf)
    await fm.send_message(message, template_name=template_name)

async def send_training_completion_email(recipients: List[EmailStr], username: str):
    await send_email(
        recipients=recipients,
        subject="Model Training Complete",
        template_name="training_completion_email.html",
        template_body={"username": username}
    )

async def send_verification_email(recipients: List[EmailStr], username: str, verification_link: str):
    await send_email(
        recipients=recipients,
        subject="Email Verification",
        template_name="verification_email.html",
        template_body={"username": username, "verification_link": verification_link}
    )
