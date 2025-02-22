from PIL import Image
import cv2
import os

base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
dir_path = os.path.join(base_dir, 'dataset/images/')
cartoon_name = 'paperino'
image_name = cartoon_name+'_rgb.jpg'
image_path = os.path.join(dir_path, image_name)
output_path = os.path.join(dir_path, cartoon_name+'.jpg')
os.makedirs(os.path.dirname(output_path), exist_ok=True)
print('Image path:', image_path)

# Options for dithering
## 1. Floyd-Steinberg: more details, but more noise
## 2. Otsu's Thresholding: less details, but better visualisation

# Choose the dithering method
otsu = True
floyd = False

if otsu:
    image = cv2.imread(image_path)
    if image is None:
        print('Error: Image not found')
        exit()
    # Resize the image to a default resolution
    default_resolution = (400, 400)  
    image = cv2.resize(image, default_resolution, interpolation=cv2.INTER_AREA)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  
    _, binary_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    #cv2.imshow('Binary Image. Press 0 to close', binary_image)
    #cv2.waitKey(0)
    cv2.imwrite(output_path, binary_image)


if floyd:

    image = Image.open(image_path).convert('L')
    image = image.resize((400, 400), Image.ANTIALIAS)
    binary_image = image.convert('1')  
    #binary_image.show()
    binary_image.save(output_path)

