from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from app.api.routes.auth import get_current_user
from app.models.user import User
from app.schemas.vision import ImageTravelInsights
from app.services.vision_service import (
    ImageTooLargeError,
    InvalidImageError,
    UnsupportedImageTypeError,
    VisionAnalysisError,
    vision_service,
)

router = APIRouter(prefix="/vision", tags=["Vision"])


@router.post(
    "/analyze",
    response_model=ImageTravelInsights,
    summary="Analyze a travel image",
    responses={
        400: {"description": "Empty file or content that is not a valid image"},
        401: {"description": "Missing or invalid bearer token"},
        413: {"description": "Image exceeds MAX_IMAGE_UPLOAD_MB"},
        415: {"description": "Unsupported image type"},
        502: {"description": "The AI model failed to analyze the image"},
    },
)
def analyze_image(
    image: UploadFile = File(..., description="JPEG, PNG, GIF, or WebP image"),
    current_user: User = Depends(get_current_user),
):
    """
    Uses Claude's vision capability to extract travel-planning information from an image:
    whether it is travel related, the likely destination and confidence, visible landmarks,
    the setting, and a suggested trip style.

    The image is analyzed once and is not stored. The structured result can be used to
    create a trip (`likely_destination` → `destination`, `suggested_trip_style` → `trip_style`)
    and then generate an itinerary.
    """
    try:
        data = image.file.read(vision_service.max_bytes + 1)
        return vision_service.analyze(data, image.content_type or "")
    except UnsupportedImageTypeError as exc:
        raise HTTPException(status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail=str(exc))
    except ImageTooLargeError as exc:
        raise HTTPException(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail=str(exc))
    except InvalidImageError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    except VisionAnalysisError as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=f"AI image analysis failed: {exc}")
