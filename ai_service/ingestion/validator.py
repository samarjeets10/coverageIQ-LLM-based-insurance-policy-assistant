import os
import hashlib
from pathlib import Path
from typing import Union, Tuple, Optional
import fitz
from ai_service.config import settings


class IngestionError(Exception):
    """Base class for all ingestion exceptions."""
    pass


class InvalidPDFSignatureError(IngestionError):
    """Raised when file lacks valid PDF magic bytes (%PDF-)."""
    pass


class CorruptedPDFError(IngestionError):
    """Raised when PDF file structure, xref table, or stream is unreadable or damaged."""
    pass


class PasswordProtectedPDFError(IngestionError):
    """Raised when PDF requires a password to open or decrypt."""
    pass


class DocumentTooLargeError(IngestionError):
    """Raised when PDF file size exceeds the allowed threshold."""
    pass


class EmptyDocumentError(IngestionError):
    """Raised when PDF has zero pages or contains no extractable page objects."""
    pass


class PathTraversalError(IngestionError):
    """Raised when PDF file path attempts directory traversal or points to an unauthorized path."""
    pass


class PDFValidator:
    """
    Validates PDF integrity, signatures, permissions, and security constraints.
    Does not trust user-supplied filenames or MIME types.
    """

    MAGIC_BYTES = b"%PDF-"
    MAGIC_SEARCH_WINDOW = 1024  # Look for %PDF- in first 1KB (accounts for UTF BOMs or header comments)

    @classmethod
    def sanitize_path(cls, file_path: Union[str, Path], allowed_root: Optional[Union[str, Path]] = None) -> Path:
        """
        Validates that a file path is safe and does not traverse outside an optional allowed root.
        """
        if not file_path:
            raise PathTraversalError("File path cannot be empty.")

        path_str = str(file_path)
        if "\0" in path_str:
            raise PathTraversalError("Null byte detected in file path.")

        resolved_path = Path(path_str).resolve()

        if allowed_root:
            resolved_root = Path(allowed_root).resolve()
            try:
                # In Python 3.9+, is_relative_to or commonpath checks traversal
                resolved_path.relative_to(resolved_root)
            except ValueError:
                raise PathTraversalError(f"Path '{resolved_path}' traverses outside allowed directory '{resolved_root}'.")

        if not resolved_path.is_file():
            raise IngestionError(f"File not found or is not a regular file: '{resolved_path}'.")

        return resolved_path

    @classmethod
    def validate_bytes(cls, data: bytes, max_size_bytes: Optional[int] = None) -> Tuple[str, int]:
        """
        Validates in-memory PDF bytes.
        Returns:
            Tuple of (sha256_hash, file_size_bytes)
        """
        max_size = max_size_bytes or settings.MAX_PDF_SIZE_BYTES
        file_size = len(data)

        if file_size == 0:
            raise EmptyDocumentError("PDF byte stream is empty (0 bytes).")

        if file_size > max_size:
            raise DocumentTooLargeError(
                f"PDF size ({file_size} bytes) exceeds maximum allowed size ({max_size} bytes)."
            )

        # Check for %PDF- magic bytes within search window
        window = data[:cls.MAGIC_SEARCH_WINDOW]
        if cls.MAGIC_BYTES not in window:
            raise InvalidPDFSignatureError(
                "Invalid PDF signature: missing '%PDF-' magic bytes in document header."
            )

        sha256 = hashlib.sha256(data).hexdigest()
        return sha256, file_size

    @classmethod
    def validate_file(
        cls,
        file_path: Union[str, Path],
        allowed_root: Optional[Union[str, Path]] = None,
        max_size_bytes: Optional[int] = None
    ) -> Tuple[Path, str, int]:
        """
        Validates a PDF on disk.
        Returns:
            Tuple of (safe_path, sha256_hash, file_size_bytes)
        """
        safe_path = cls.sanitize_path(file_path, allowed_root=allowed_root)
        max_size = max_size_bytes or settings.MAX_PDF_SIZE_BYTES

        file_size = safe_path.stat().st_size
        if file_size == 0:
            raise EmptyDocumentError(f"PDF file is empty (0 bytes): '{safe_path}'.")

        if file_size > max_size:
            raise DocumentTooLargeError(
                f"PDF size ({file_size} bytes) exceeds maximum allowed size ({max_size} bytes): '{safe_path}'."
            )

        # Read first window to check magic bytes
        with open(safe_path, "rb") as f:
            header = f.read(cls.MAGIC_SEARCH_WINDOW)
            if cls.MAGIC_BYTES not in header:
                raise InvalidPDFSignatureError(
                    f"File '{safe_path}' does not contain valid '%PDF-' header signature."
                )

            # Compute SHA-256
            f.seek(0)
            hasher = hashlib.sha256()
            while chunk := f.read(65536):
                hasher.update(chunk)
            sha256 = hasher.hexdigest()

        # Structural inspection using fitz
        try:
            with fitz.open(safe_path) as doc:
                if doc.needs_pass or doc.is_encrypted:
                    raise PasswordProtectedPDFError(
                        f"PDF document is password-protected or encrypted: '{safe_path}'."
                    )
                if len(doc) == 0:
                    raise EmptyDocumentError(f"PDF document contains 0 pages: '{safe_path}'.")
        except PasswordProtectedPDFError:
            raise
        except EmptyDocumentError:
            raise
        except Exception as e:
            raise CorruptedPDFError(f"Corrupted or unreadable PDF document '{safe_path}': {e}") from e

        return safe_path, sha256, file_size
