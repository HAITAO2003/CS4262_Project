"""
Data validation schemas for the Chat Engine API.

This module provides the Pydantic models used to parse, validate, and serialize
incoming HTTP requests and outgoing responses for the chat completions endpoint.
"""

from pydantic import BaseModel


class ChatMessage(BaseModel):
    """
    Represents a single message in the chat history.
    """
    role: str
    content: str


class ChatRequest(BaseModel):
    """
    Represents an incoming chat completion request.
    """
    messages: list[ChatMessage]
    temperature: float = 0.7
    max_tokens: int = 128
    priority: int | None = None


class ChatResponse(BaseModel):
    """
    Represents an outgoing chat completion response.
    """
    output: str
    logprobs: list[float]
