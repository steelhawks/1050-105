from PIL import Image
from io import BytesIO

def convert_to_jpg(image):
    """TBW."""
    # assumes image is RGB provided
    im = Image.fromarray(image)
    mem_file = BytesIO()
    im.save(mem_file, 'JPEG')
    return mem_file.getvalue()
