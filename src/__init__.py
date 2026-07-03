"""duckbot-rag-memory: persistent RAG + memory layer for OpenClaw/Hermes."""

__version__ = "0.15.2"


# ---------------------------------------------------------------------------
# Posthog 7.x API fix — MUST be applied before chromadb is imported.
# Posthog 7.x changed capture() from:
#   capture(user_id, event_name, properties={})
# to:
#   capture(event_name, properties={})
# ChromaDB's telemetry code was written for posthog 2.x and calls the old
# API, causing "capture() takes 1 positional argument but 3 were given" on
# every chromadb operation. This patch fixes _direct_capture() before any
# chromadb code runs.
# ---------------------------------------------------------------------------
def _patch_posthog() -> None:
    try:
        import chromadb.telemetry.product.posthog as _ph_module
        _original = getattr(_ph_module.Posthog, '_direct_capture', None)
        if _original is None:
            return
        # Check if already patched
        if getattr(_original, '_posthog7_patched', False):
            return

        def _patched(self, event) -> None:
            try:
                import posthog as _ph
                settings = getattr(_ph_module, 'POSTHOG_EVENT_SETTINGS', {})
                ctx = getattr(self, 'context', {})
                props = {**getattr(event, 'properties', {}), **settings, **ctx}
                _ph.capture(getattr(event, 'name', str(event)), properties=props)
            except Exception:
                pass  # telemetry is non-critical — never crash over this

        _patched._posthog7_patched = True  # type: ignore[attr-defined]
        _ph_module.Posthog._direct_capture = _patched
    except Exception:
        pass


_patch_posthog()
del _patch_posthog


# ---------------------------------------------------------------------------
# ChromaDB logging suppression.
# ChromaDB 0.5.x logs "Delete of nonexisting embedding ID" as WARNING
# via the standard `logging` module. Python's lastResort handler sends
# WARNING+ to stderr when no handlers are configured, polluting stdout
# (which subprocess callers capture as the "result"). Attaching a
# NullHandler suppresses this without affecting any real error handling.
# ---------------------------------------------------------------------------
import logging

_chroma_log = logging.getLogger("chromadb")
_chroma_log.setLevel(logging.WARNING)
_chroma_log.addHandler(logging.NullHandler())
