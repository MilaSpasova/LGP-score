from .openai_client import simplify_with_openai

__all__ = ["simplify_with_openai"]

try:  # Optional until Gemini integration is added
    from .gemini_client import simplify_with_gemini

    __all__.append("simplify_with_gemini")
except ModuleNotFoundError:
    pass

