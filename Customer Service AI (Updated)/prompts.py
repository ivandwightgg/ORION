# app/prompts.py
"""
Customer support agent prompt templates for multi-tier AI system.

This module defines system and user prompts for a two-tier customer support
architecture where Tier 1 provides quick JSON responses and Tier 2 handles
complex queries requiring deeper reasoning.

Compatible with Python 3.8+
"""

from typing import Final, Optional, Dict, Any, List
from pydantic import BaseModel, Field, validator, ValidationError
from enum import Enum
import json


# ============================================================================
# PYDANTIC MODELS FOR RESPONSE VALIDATION
# ============================================================================

class Tier1Response(BaseModel):
    """
    Validated schema for Tier 1 agent JSON responses.
    
    Attributes:
        answer: The support agent's response to the user
        confidence: Confidence score between 0.0 and 1.0
        needs_web: Whether web search is needed for better answer
        reasons: List of reasons explaining the confidence/needs_web values
    """
    answer: str = Field(..., min_length=1, description="The agent's response")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score 0.0-1.0")
    needs_web: bool = Field(..., description="Whether web search is needed")
    reasons: List[str] = Field(..., min_items=1, description="Reasoning for confidence/needs_web")
    
    @validator('reasons')
    def validate_reasons(cls, v):
        """Ensure all reasons are non-empty strings."""
        if not all(isinstance(r, str) and r.strip() for r in v):
            raise ValueError("All reasons must be non-empty strings")
        return v
    
    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "answer": "To reset your password, click 'Forgot Password' on the login page.",
                "confidence": 0.95,
                "needs_web": False,
                "reasons": ["Clear instructions found in documentation"]
            }
        }


class ConfidenceLevel(str, Enum):
    """Enumeration of confidence levels for easier categorization."""
    HIGH = "high"  # 0.8 - 1.0
    MEDIUM = "medium"  # 0.5 - 0.79
    LOW = "low"  # 0.0 - 0.49


# ============================================================================
# PROMPT CONSTANTS
# ============================================================================

TIER1_SYSTEM: Final[str] = """
You are a concise, friendly customer support agent.

Rules:
- Use the provided company/documentation context when it is relevant.
- If the user is greeting or making small talk and context is not needed, you may respond conversationally.
- Always return a valid JSON object ONLY (no extra text before or after), with these exact keys:
  {
    "answer": "<short helpful answer>",
    "confidence": <float between 0.0 and 1.0>,
    "needs_web": <boolean true or false>,
    "reasons": ["brief reason 1", "brief reason 2"]
  }
- The "answer" field should contain your response to the user.
- The "confidence" field represents how confident you are in your answer (0.0 = no confidence, 1.0 = complete confidence).
- Set "needs_web" to true if you need additional web research to provide a complete answer.
- The "reasons" array should explain why you set needs_web to true, or why confidence is low.
- Do NOT include any explanatory text, markdown formatting, or code blocks around the JSON.
- Do NOT reveal your reasoning process or chain-of-thought in the response.
"""

TIER1_USER_TEMPLATE: Final[str] = """
User said:
{user_message}

Relevant context (from internal docs):
{context}

Instructions: Return ONLY a valid JSON object with no additional text, markdown, or formatting.
"""

TIER2_SYSTEM: Final[str] = """
You are an advanced reasoning agent providing in-depth customer support.

Guidelines:
- Think through the problem carefully, but do NOT reveal your chain-of-thought or internal reasoning to the user.
- You may receive web search results prepared by the orchestrator to supplement your knowledge.
- Synthesize information from all available sources (internal context, web results, and your knowledge) into a clear, accurate, and comprehensive answer.
- If you reference external sources, cite them using bracketed numbers [1], [2], etc., which correspond to the sources list displayed to the user.
- Use clear language and step-by-step explanations when appropriate.
- If the query involves critical decisions (medical, legal, financial, safety-related), include this disclaimer at the end: "For this specific question, please also consult a qualified professional."
- Structure your response with appropriate paragraphs for readability.
- Be thorough but concise - provide enough detail to be helpful without overwhelming the user.

Return only your final answer as plain text prose (no JSON formatting required for Tier 2).
"""

TIER2_USER_TEMPLATE: Final[str] = """
User message:
{user_message}

Retrieved internal context:
{retrieved_context}

Tier-1 assessment:
{difficulty_report}

Web search results (if available):
{web_context}

Task: Provide a comprehensive, well-reasoned answer that addresses the user's question completely. Use clear language and cite sources when referencing external information.
"""


# ============================================================================
# SYNCHRONOUS HELPER FUNCTIONS
# ============================================================================

def format_tier1_prompt(user_message: str, context: str) -> str:
    """
    Format the Tier 1 user prompt with proper escaping.
    
    Args:
        user_message: The user's input message
        context: Retrieved context from internal documentation
        
    Returns:
        Formatted prompt string ready for LLM
        
    Raises:
        ValueError: If required parameters are None or empty
    """
    if not user_message or not isinstance(user_message, str):
        raise ValueError("user_message must be a non-empty string")
    if context is None:
        raise ValueError("context must not be None (use empty string if no context)")
    
    return TIER1_USER_TEMPLATE.format(
        user_message=user_message.strip(),
        context=context.strip() if context else "No relevant context available."
    )


def format_tier2_prompt(
    user_message: str,
    retrieved_context: str,
    difficulty_report: str,
    web_context: str = ""
) -> str:
    """
    Format the Tier 2 user prompt with proper escaping.
    
    Args:
        user_message: The user's input message
        retrieved_context: Context from internal documentation
        difficulty_report: Assessment from Tier 1 agent
        web_context: Optional web search results (default: empty string)
        
    Returns:
        Formatted prompt string ready for LLM
        
    Raises:
        ValueError: If required parameters are invalid
    """
    if not user_message or not isinstance(user_message, str):
        raise ValueError("user_message must be a non-empty string")
    if retrieved_context is None:
        raise ValueError("retrieved_context must not be None")
    if difficulty_report is None:
        raise ValueError("difficulty_report must not be None")
    
    return TIER2_USER_TEMPLATE.format(
        user_message=user_message.strip(),
        retrieved_context=retrieved_context.strip() if retrieved_context else "No context available.",
        difficulty_report=difficulty_report.strip() if difficulty_report else "No assessment available.",
        web_context=web_context.strip() if web_context else "No web results available."
    )


def parse_tier1_response(response_text: str) -> Tier1Response:
    """
    Parse and validate a Tier 1 JSON response.
    
    Args:
        response_text: Raw text response from Tier 1 LLM
        
    Returns:
        Validated Tier1Response object
        
    Raises:
        ValidationError: If response doesn't match schema
        json.JSONDecodeError: If response isn't valid JSON
    """
    # Strip markdown code blocks if present
    cleaned_text = response_text.strip()
    if cleaned_text.startswith("```json"):
        cleaned_text = cleaned_text[7:]
    if cleaned_text.startswith("```"):
        cleaned_text = cleaned_text[3:]
    if cleaned_text.endswith("```"):
        cleaned_text = cleaned_text[:-3]
    
    cleaned_text = cleaned_text.strip()
    
    # Parse JSON
    response_dict = json.loads(cleaned_text)
    
    # Validate with Pydantic
    return Tier1Response(**response_dict)


def get_confidence_level(confidence: float) -> ConfidenceLevel:
    """
    Categorize a confidence score into a level.
    
    Args:
        confidence: Float between 0.0 and 1.0
        
    Returns:
        ConfidenceLevel enum value
    """
    if confidence >= 0.8:
        return ConfidenceLevel.HIGH
    elif confidence >= 0.5:
        return ConfidenceLevel.MEDIUM
    else:
        return ConfidenceLevel.LOW


# ============================================================================
# ASYNC HELPER FUNCTIONS
# ============================================================================

async def format_tier1_prompt_async(user_message: str, context: str) -> str:
    """
    Async version of format_tier1_prompt for use in async applications.
    
    Args:
        user_message: The user's input message
        context: Retrieved context from internal documentation
        
    Returns:
        Formatted prompt string ready for LLM
        
    Raises:
        ValueError: If required parameters are None or empty
    """
    return format_tier1_prompt(user_message, context)


async def format_tier2_prompt_async(
    user_message: str,
    retrieved_context: str,
    difficulty_report: str,
    web_context: str = ""
) -> str:
    """
    Async version of format_tier2_prompt for use in async applications.
    
    Args:
        user_message: The user's input message
        retrieved_context: Context from internal documentation
        difficulty_report: Assessment from Tier 1 agent
        web_context: Optional web search results
        
    Returns:
        Formatted prompt string ready for LLM
        
    Raises:
        ValueError: If required parameters are invalid
    """
    return format_tier2_prompt(user_message, retrieved_context, difficulty_report, web_context)


async def parse_tier1_response_async(response_text: str) -> Tier1Response:
    """
    Async version of parse_tier1_response.
    
    Args:
        response_text: Raw text response from Tier 1 LLM
        
    Returns:
        Validated Tier1Response object
        
    Raises:
        ValidationError: If response doesn't match schema
        json.JSONDecodeError: If response isn't valid JSON
    """
    return parse_tier1_response(response_text)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def create_sample_tier1_response() -> Dict[str, Any]:
    """
    Create a sample Tier 1 response for testing.
    
    Returns:
        Dictionary with sample response data
    """
    return {
        "answer": "To reset your password, click the 'Forgot Password' link on the login page.",
        "confidence": 0.95,
        "needs_web": False,
        "reasons": ["Clear documentation found", "Common procedure"]
    }


def validate_tier1_json_string(json_string: str) -> tuple[bool, Optional[str]]:
    """
    Validate a JSON string against Tier 1 schema without raising exceptions.
    
    Args:
        json_string: JSON string to validate
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        parse_tier1_response(json_string)
        return True, None
    except ValidationError as e:
        return False, f"Validation error: {str(e)}"
    except json.JSONDecodeError as e:
        return False, f"JSON decode error: {str(e)}"
    except Exception as e:
        return False, f"Unexpected error: {str(e)}"


# ============================================================================
# EXAMPLE USAGE AND TESTS
# ============================================================================

if __name__ == "__main__":
    import asyncio
    
    print("=" * 70)
    print("PROMPT SYSTEM DEMONSTRATION")
    print("=" * 70)
    
    # Test 1: Tier 1 Prompt Formatting
    print("\n[TEST 1] Tier 1 Prompt Formatting")
    print("-" * 70)
    try:
        tier1_prompt = format_tier1_prompt(
            user_message="How do I reset my password?",
            context="Password reset: Click 'Forgot Password' on login page, enter email, check inbox for reset link."
        )
        print("✓ Success!")
        print(f"Prompt length: {len(tier1_prompt)} characters")
    except Exception as e:
        print(f"✗ Failed: {e}")
    
    # Test 2: Tier 1 Response Validation
    print("\n[TEST 2] Tier 1 Response Validation")
    print("-" * 70)
    sample_response = json.dumps(create_sample_tier1_response())
    print(f"Sample JSON: {sample_response}")
    
    try:
        validated = parse_tier1_response(sample_response)
        print("✓ Validation successful!")
        print(f"  Answer: {validated.answer}")
        print(f"  Confidence: {validated.confidence}")
        print(f"  Confidence Level: {get_confidence_level(validated.confidence).value}")
        print(f"  Needs Web: {validated.needs_web}")
        print(f"  Reasons: {', '.join(validated.reasons)}")
    except Exception as e:
        print(f"✗ Validation failed: {e}")
    
    # Test 3: Invalid JSON Handling
    print("\n[TEST 3] Invalid JSON Handling")
    print("-" * 70)
    invalid_json = '{"answer": "test", "confidence": 1.5}'  # confidence > 1.0
    is_valid, error = validate_tier1_json_string(invalid_json)
    if not is_valid:
        print(f"✓ Correctly rejected invalid JSON")
        print(f"  Error: {error}")
    else:
        print("✗ Failed to reject invalid JSON")
    
    # Test 4: Markdown Stripping
    print("\n[TEST 4] Markdown Code Block Stripping")
    print("-" * 70)
    markdown_json = f"```json\n{sample_response}\n```"
    try:
        validated = parse_tier1_response(markdown_json)
        print("✓ Successfully stripped markdown and validated!")
    except Exception as e:
        print(f"✗ Failed: {e}")
    
    # Test 5: Tier 2 Prompt Formatting
    print("\n[TEST 5] Tier 2 Prompt Formatting")
    print("-" * 70)
    try:
        tier2_prompt = format_tier2_prompt(
            user_message="What's your refund policy?",
            retrieved_context="Full refund within 30 days for defective items.",
            difficulty_report='{"confidence": 0.7, "needs_web": true}',
            web_context="Consumer protection laws require clear refund policies."
        )
        print("✓ Success!")
        print(f"Prompt length: {len(tier2_prompt)} characters")
    except Exception as e:
        print(f"✗ Failed: {e}")
    
    # Test 6: Async Functions
    print("\n[TEST 6] Async Function Testing")
    print("-" * 70)
    
    async def test_async():
        try:
            prompt = await format_tier1_prompt_async(
                user_message="Test message",
                context="Test context"
            )
            print("✓ Async Tier 1 formatting works!")
            
            prompt2 = await format_tier2_prompt_async(
                user_message="Test",
                retrieved_context="Context",
                difficulty_report="Report"
            )
            print("✓ Async Tier 2 formatting works!")
            
            response = await parse_tier1_response_async(sample_response)
            print("✓ Async parsing works!")
            
        except Exception as e:
            print(f"✗ Async test failed: {e}")
    
    asyncio.run(test_async())
    
    # Test 7: Error Handling
    print("\n[TEST 7] Error Handling")
    print("-" * 70)
    try:
        format_tier1_prompt("", "context")
        print("✗ Should have raised ValueError for empty user_message")
    except ValueError as e:
        print(f"✓ Correctly raised ValueError: {e}")
    
    try:
        format_tier2_prompt("message", None, "report")
        print("✗ Should have raised ValueError for None retrieved_context")
    except ValueError as e:
        print(f"✓ Correctly raised ValueError: {e}")
    
    print("\n" + "=" * 70)
    print("ALL TESTS COMPLETED")
    print("=" * 70)