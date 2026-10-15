"""Pins the content hash of a tiny fixture so any change to the procedure is deliberate."""

from datacopilot.core.schema import Annotation, Category, Image
from datacopilot.core.version import models_version

GOLDEN_VERSION = "276ca0bc64ebb0bc22f12ebfedbc6c40"


def test_golden_version_is_stable_and_order_independent() -> None:
    images = [
        Image(
            image_id="0123456789abcdef", file_path="/x/a.png", width=100, height=50, split="train"
        ),
        Image(image_id="fedcba9876543210", file_path="/x/b.png", width=64, height=64),
    ]
    anns = [
        Annotation(
            ann_id="1",
            image_id="0123456789abcdef",
            category_id=3,
            kind="bbox",
            bbox_xyxy_abs=(10.5, 20.0, 60.25, 40.0),
            is_crowd=True,
            source={"format": "coco", "id": 1},
        ),
    ]
    cats = [Category(id=3, name="cat", attributes={"supercategory": "animal"})]
    assert models_version(images, anns, cats, schema_version=1) == GOLDEN_VERSION
    assert models_version(images[::-1], anns[::-1], cats[::-1], schema_version=1) == GOLDEN_VERSION
