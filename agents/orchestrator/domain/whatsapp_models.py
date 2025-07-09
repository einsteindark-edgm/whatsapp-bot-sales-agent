"""
WhatsApp Business API message models.

This module defines the Pydantic models for WhatsApp webhook payloads
and API responses following the official Meta API structure.
"""

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List


class WhatsAppMetadata(BaseModel):
    """WhatsApp message metadata containing phone information."""

    display_phone_number: str = Field(..., description="Display phone number")
    phone_number_id: str = Field(..., description="Phone number ID")


class WhatsAppTextMessage(BaseModel):
    """WhatsApp text message content."""

    body: str = Field(..., description="Message text body")


class WhatsAppMessage(BaseModel):
    """Individual WhatsApp message from webhook."""

    from_: str = Field(..., alias="from", description="Sender phone number")
    id: str = Field(..., description="Message ID")
    timestamp: str = Field(..., description="Message timestamp")
    type: str = Field(..., description="Message type (text, image, etc.)")
    text: Optional[WhatsAppTextMessage] = Field(None, description="Text message content")


class WhatsAppContact(BaseModel):
    """WhatsApp contact information."""

    profile: Dict[str, str] = Field(..., description="Contact profile information")
    wa_id: str = Field(..., description="WhatsApp ID")


class WhatsAppValue(BaseModel):
    """WhatsApp webhook change value containing message data."""

    messaging_product: str = Field(..., description="Always 'whatsapp'")
    metadata: WhatsAppMetadata = Field(..., description="Message metadata")
    messages: Optional[List[WhatsAppMessage]] = Field(None, description="List of messages")
    contacts: Optional[List[WhatsAppContact]] = Field(None, description="List of contacts")
    statuses: Optional[List[Dict[str, Any]]] = Field(None, description="Message statuses")


class WhatsAppChange(BaseModel):
    """WhatsApp webhook change event."""

    value: WhatsAppValue = Field(..., description="Change value")
    field: str = Field(..., description="Field type (messages, statuses, etc.)")


class WhatsAppEntry(BaseModel):
    """WhatsApp webhook entry containing changes."""

    id: str = Field(..., description="WhatsApp Business Account ID")
    changes: List[WhatsAppChange] = Field(..., description="List of changes")


class WhatsAppWebhookPayload(BaseModel):
    """Complete WhatsApp webhook payload structure."""

    object: str = Field(..., description="Always 'whatsapp_business_account'")
    entry: List[WhatsAppEntry] = Field(..., description="List of entries")

    def extract_message(self) -> Optional[tuple[str, str, str]]:
        """
        Extract message text, sender, and message_id from webhook payload.

        Returns:
            Optional[tuple[str, str, str]]: (message_text, sender_phone, message_id) or None
        """
        try:
            # Navigate through the nested structure
            for entry in self.entry:
                for change in entry.changes:
                    if change.field == "messages" and change.value.messages:
                        # Get the first message (usually only one per webhook)
                        message = change.value.messages[0]
                        if message.type == "text" and message.text:
                            return (
                                message.text.body,
                                message.from_,
                                message.id,
                            )
        except (IndexError, AttributeError):
            pass
        return None

    def extract_status_update(self) -> Optional[Dict[str, Any]]:
        """
        Extract status update information from webhook payload.

        Returns:
            Optional[Dict[str, Any]]: Status update information or None
        """
        try:
            for entry in self.entry:
                for change in entry.changes:
                    if change.field == "messages" and change.value.statuses:
                        # Get the first status update
                        status = change.value.statuses[0]
                        return status
        except (IndexError, AttributeError):
            pass
        return None

    def get_phone_number_id(self) -> Optional[str]:
        """
        Extract phone number ID from webhook payload.

        Returns:
            Optional[str]: Phone number ID or None
        """
        try:
            for entry in self.entry:
                for change in entry.changes:
                    if change.value.metadata:
                        return change.value.metadata.phone_number_id
        except (IndexError, AttributeError):
            pass
        return None


# WhatsApp API Request Models


class WhatsAppTextPayload(BaseModel):
    """Text payload for sending messages."""

    body: str = Field(..., description="Message text", max_length=4096)
    preview_url: bool = Field(default=False, description="Enable URL preview")


class WhatsAppSendMessageRequest(BaseModel):
    """Request model for sending WhatsApp messages."""

    messaging_product: str = Field(default="whatsapp", description="Always 'whatsapp'")
    recipient_type: str = Field(default="individual", description="Recipient type")
    to: str = Field(..., description="Recipient phone number")
    type: str = Field(default="text", description="Message type")
    text: WhatsAppTextPayload = Field(..., description="Text message content")


class WhatsAppMarkReadRequest(BaseModel):
    """Request model for marking messages as read."""

    messaging_product: str = Field(default="whatsapp", description="Always 'whatsapp'")
    status: str = Field(default="read", description="Status to set")
    message_id: str = Field(..., description="Message ID to mark as read")


class WhatsAppTypingIndicatorRequest(BaseModel):
    """Request model for sending typing indicator."""

    messaging_product: str = Field(default="whatsapp", description="Always 'whatsapp'")
    recipient_type: str = Field(default="individual", description="Recipient type")
    to: str = Field(..., description="Recipient phone number")
    typing: str = Field(default="on", description="Typing status (on/off)")


# WhatsApp API Response Models


class WhatsAppMessageResponse(BaseModel):
    """Response model for sent WhatsApp messages."""

    messaging_product: str = Field(..., description="Always 'whatsapp'")
    contacts: List[Dict[str, str]] = Field(..., description="Contact information")
    messages: List[Dict[str, str]] = Field(..., description="Sent message information")


class WhatsAppErrorDetail(BaseModel):
    """WhatsApp API error detail."""

    message: str = Field(..., description="Error message")
    type: str = Field(..., description="Error type")
    code: int = Field(..., description="Error code")
    fbtrace_id: Optional[str] = Field(None, description="Facebook trace ID")


class WhatsAppErrorResponse(BaseModel):
    """WhatsApp API error response."""

    error: WhatsAppErrorDetail = Field(..., description="Error details")