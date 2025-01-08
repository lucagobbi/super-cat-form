from enum import Enum
from datetime import datetime
from typing import Dict, Any, List, Callable

from pydantic import BaseModel

from cat.log import log


class FormEvent(Enum):

    # Lifecycle events
    FORM_INITIALIZED = "form_initialized"
    FORM_SUBMITTED = "form_submitted"
    FORM_CLOSED = "form_closed"

    # Extraction events
    EXTRACTION_STARTED = "extraction_started"
    EXTRACTION_COMPLETED = "extraction_completed"

    # Validation events
    VALIDATION_STARTED = "validation_started"
    VALIDATION_COMPLETED = "validation_completed"

    FIELD_UPDATED = "field_updated"

    # Tool events
    TOOL_STARTED = "tool_started"
    TOOL_EXECUTED = "tool_executed"


class FormEventContext(BaseModel):
    timestamp: datetime
    form_id: str
    event: FormEvent
    data: Dict[str, Any]


class FormEventManager:
    def __init__(self):
        self._handlers: Dict[FormEvent, List[Callable[[FormEventContext], None]]] = {
            event: [] for event in FormEvent
        }

    def on(self, event: FormEvent, handler: Callable[[FormEventContext], None]):
        """Register an event handler"""
        self._handlers[event].append(handler)

    def emit(self, event: FormEvent, data: Dict[str, Any], form_id: str):
        """Emit an event to all registered handlers"""
        context = FormEventContext(
            timestamp=datetime.now(),
            form_id=form_id,
            event=event,
            data=data
        )

        for handler in self._handlers[event]:
            try:
                handler(context)
            except Exception as e:
                log.error(f"Error in event handler: {str(e)}")