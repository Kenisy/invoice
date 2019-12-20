import utils
import cv2
import os
import json
import constant

if __name__ == "__main__":  
    imgurl = 'input\\6c380ab17f41c2ff591c9ff91194469f.png'
    img = cv2.imread(imgurl)
    img = utils.preprocess_img(img)

    # get bounding box of all rectangles
    bounding_boxes = utils.box_extraction(img)

    table = utils.get_item_table(bounding_boxes, img)

    # write output to json
    base = os.path.basename(imgurl)
    output = constant.OUTPUT_PATH + os.path.splitext(base)[0] + '.json'
    with open(output, 'w+') as fp:
        json.dump(table, fp, indent=4)