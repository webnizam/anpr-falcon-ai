
# from paddleocr import PaddleOCR,draw_ocr
import cv2
import easyocr


# ocr = PaddleOCR(lang='en') # need to run only once to download and load model into memory
img_path = './test_images/lpd (1).jpg'

ocr = easyocr.Reader(['en'])

result = ocr.readtext(img_path)

# draw result
result = result[0]
image = cv2.imread(img_path)
# boxes = [line[0] for line in result]
# txts = [line[1] for line in result]
# scores = [line[1][1] for line in result]
font = cv2.FONT_HERSHEY_SIMPLEX
org = (50, 50)
fontScale = 1
color = (255, 0, 0)
thickness = 2

print(str(result))
# im_show = draw_ocr(image, boxes, txts, scores, font_path='simfang.ttf')
im_show = cv2.putText(image, str(result), org, font,
                      fontScale, color, thickness, cv2.LINE_AA)
# im_show = Image.fromarray(im_show)
cv2.imwrite('result.jpg', im_show)
