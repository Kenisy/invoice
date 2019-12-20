import utils
import cv2
import os
import json
import constant

if __name__ == "__main__":  
    imgurl = 'input\\Alter-DHL-Express-Commercial-Invoice-Title-Sample-Commercial-Invoice.png'
    img = cv2.imread(imgurl)
    img = utils.preprocess_img(img)

    # get bounding box of all rectangles
    bounding_boxes = utils.box_extraction(img)
    print(f'[INFO] bounding_boxes: {len(bounding_boxes)}')

    # get invoice data
    data = utils.get_invoice_data(bounding_boxes, img)

    # write output to json
    base = os.path.basename(imgurl)
    output = constant.OUTPUT_PATH + os.path.splitext(base)[0] + '.json'
    with open(output, 'w+') as fp:
        json.dump(data, fp, indent=4)
    print(f'[INFO] exported to {output}')