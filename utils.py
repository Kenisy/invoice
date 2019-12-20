import numpy as np
import pytesseract
import cv2
import os
import constant

PREPROCESS_PATH = constant.PREP_PATH

def remove_parent_contours(contours, hierarchy):
  result = []
  for idx, hier in enumerate(hierarchy[0]):
    if hier[2] == -1:
      result.append(contours[idx])
  return result

def sort_contours(cnts, image, method="left-to-right"):
    # initialize the reverse flag and sort index
    reverse = False
    i = 0

    # handle if we need to sort in reverse
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True

    # handle if we are sorting against the y-coordinate rather than
    # the x-coordinate of the bounding box
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1

    # construct the list of bounding boxes and sort them from top to
    # bottom
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
    key=lambda b:b[1][0] + b[1][1] * image.shape[1], reverse=reverse))

    # return the list of sorted contours and bounding boxes
    return (cnts, boundingBoxes)
 
def box_extraction(image):
    origin = image.copy()
    img = image.copy()
    (thresh, img_bin) = cv2.threshold(img, 128, 255,
                                        cv2.THRESH_BINARY | cv2.THRESH_OTSU)  # Thresholding the image
    img_bin = 255-img_bin  # Invert the image
    cv2.imwrite(PREPROCESS_PATH + "Image_bin.jpg",img_bin)

    # Defining a kernel length
    vertical_kernel_length = np.array(img).shape[1] // 80
    horizontal_kernel_length = np.array(img).shape[0] // 80
        
    # A verticle kernel of (1 X kernel_length), which will detect all the verticle lines from the image.
    verticle_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vertical_kernel_length))
    # A horizontal kernel of (kernel_length X 1), which will help to detect all the horizontal line from the image.
    hori_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_kernel_length, 1))
    # A kernel of (3 X 3) ones.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    # Morphological operation to detect verticle lines from an image
    img_temp1 = cv2.erode(img_bin, verticle_kernel, iterations=2)
    cv2.imwrite(PREPROCESS_PATH + "img_temp1_lines.jpg",img_temp1)
    verticle_lines_img = cv2.dilate(img_temp1, verticle_kernel, iterations=3)
    cv2.imwrite(PREPROCESS_PATH + "verticle_lines.jpg",verticle_lines_img)
    # Morphological operation to detect horizontal lines from an image
    img_temp2 = cv2.erode(img_bin, hori_kernel, iterations=3)
    horizontal_lines_img = cv2.dilate(img_temp2, hori_kernel, iterations=4)
    cv2.imwrite(PREPROCESS_PATH + "horizontal_lines.jpg",horizontal_lines_img)
    # Weighting parameters, this will decide the quantity of an image to be added to make a new image.
    alpha = 0.5
    beta = 1.0 - alpha
    # This function helps to add two image with specific weight parameter to get a third image as summation of two image.
    img_final_bin = cv2.addWeighted(verticle_lines_img, alpha, horizontal_lines_img, beta, 0.0)
    img_final_bin = cv2.erode(~img_final_bin, kernel, iterations=1)
    (thresh, img_final_bin) = cv2.threshold(img_final_bin, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # For Debugging
    # Enable this line to see verticle and horizontal lines in the image which is used to find boxes
    cv2.imwrite(PREPROCESS_PATH + "img_final_bin.jpg",img_final_bin)
    # Find contours for image, which will detect all the boxes
    contours, hierarchy = cv2.findContours(img_final_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = remove_parent_contours(contours, hierarchy)
    # Sort all the contours by top to bottom.
    (contours, boundingBoxes) = sort_contours(contours, img, method="top-to-bottom")
    idx = 0
    bounding_boxes = []
    for c in contours:
        # Returns the location and width,height for every contour
        x, y, w, h = cv2.boundingRect(c)
        # If the box height is greater then 20, widht is >80, then only save it as a box in "cropped/" folder.
        if (w > img.shape[1] / 20 and h > img.shape[0] / 100 and h < img.shape[0] * 0.8):
            idx += 1
            origin = cv2.rectangle(origin, (x, y), (x+w, y+h), (0, 0, 255), 1)
            origin = cv2.putText(origin, "#{}".format(idx-1), (x-10, y-10), cv2.FONT_HERSHEY_SIMPLEX,
            0.7, (0, 0, 255), 2)
            bounding_boxes.append((x, y, w, h))
    cv2.imwrite(PREPROCESS_PATH + "img_boxes.jpg", origin)
    print(idx)
    return bounding_boxes

def preprocess_img(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_LANCZOS4)
    # img = cv2.adaptiveThreshold(img.astype(np.uint8), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 39, 39)  
    # kernel = np.ones((1, 1), np.uint8)    
    # img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    return img

def get_text_from_img(img, preprocess=True, lang='eng', psm=6):
    # if preprocess:
    #   img = preprocess_img(img)
    config = (f'-l {lang} --oem 1 --psm {psm}')
    text = pytesseract.image_to_string(img, config=config)
    return text

def get_text_from_box(img, box, preprocess=True):
    return get_text_from_img(img[box[1]:box[1]+box[3], box[0]:box[0]+box[2]], preprocess=preprocess)

def get_data_from_img(image):
    # image = preprocess_img(image)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    config = (f'-l eng --oem 1 --psm 6')
    data = pytesseract.image_to_data(image, config=config, output_type=pytesseract.Output.DATAFRAME)
    return data

def is_row_item_header(row, img):
    keywords_found = 0
    for keyword in constant.TABLE_HEADERS:
        if any([keyword in get_text_from_box(img, box).lower() for box in row]):
            keywords_found += 1
    if keywords_found > 1:
        return True, []
    return False, []


def is_row_belong_to_header(row, header_row):
    print(row)
    print(header_row)
    if len(row) != len(header_row):
        return False
    if all(
        [
            abs(header_row[i][0] - row[i][0]) < 5
            and abs(header_row[i][2] - row[i][2]) < 5
            for i in range(len(header_row))
        ]
    ):
        return True
    return False


def is_box_belong_to_header(box, header_row):
    x_first, y_first, w_first, h_first = header_row[0]
    x_last, y_last, w_last, h_last = header_row[-1]
    x, y, w, h = box
    if abs(y - (y_first + h_first)) > 10:
        return False
    if abs(x - x_first) < 5 and abs(w - (x_last + w_last - x_first)) < 5:
        return True
    return False


def get_header_text(text):
    text = text.replace("\n", " ")
    text.strip()
    return text


def split_row(item_row, img):
    x, y, w, h = item_row[0]
    data = get_data_from_img(img[y : y + h, x : x + w])
    nlines = data["line_num"].max()
    item_rows = []
    print(f"nlines: {nlines}")
    if nlines > 1:
        prev_y = y
        for i in range(2, nlines + 1):
            line = data[data["line_num"] == i]
            line_h = line.iloc[0]["top"] - 1
            row = []
            for col in item_row:
                x, _, w, _ = col
                row.append((x, prev_y, w, y + line_h - prev_y))
            item_rows.append(row)
            prev_y = y + line_h
        row = []
        for col in item_row:
            x, _, w, _ = col
            row.append((x, prev_y, w, y + h - prev_y))
        item_rows.append(row)
        return item_rows
    else:
        return [item_row]


def split_col(item_row, header_row):
    cols = []
    _, y, _, h = item_row
    for col in header_row:
        x, _, w, _ = col
        cols.append((x, y, w, h))
    return cols

def extract_from_text(text, exporter_found, consignee_found):
    header, content = '', ''
    idx = text.find('\n')
    if exporter_found == False and any([h in text[:idx].lower() for h in constant.EXPORTER_HEADERS]):
        header = 'EXPOTER'
        content = text[idx+1:].strip()
        exporter_found = True
        print(f'exporter: {text}')
    elif consignee_found == False and any([h in text[:idx].lower() for h in constant.CONSIGNEE_HEADERS]):
        header = 'CONSIGNEE'
        content = text[idx+1:].strip()
        consignee_found = True
        print(f'consignee: {text}')
    return header, content, exporter_found, consignee_found

def get_item_table(bounding_boxes, img):
    invoice = {}
    current_row = []
    current_row_text = []
    header_row = []
    header_texts = []
    item_rows = []
    bounding_box_texts = [''] * len(bounding_boxes)
    header_row_found = False
    items_row_found = False
    consignee_found = False
    exporter_found = False
    for i in range(len(bounding_boxes) - 1):
        box = bounding_boxes[i]
        next_box = bounding_boxes[i + 1]
        box_text = get_text_from_box(img, box)
        bounding_box_texts[i] = box_text
        if exporter_found == False or consignee_found == False:
            header, content, exporter_found, consignee_found = extract_from_text(box_text, exporter_found, consignee_found)
            if header != '' and content != '':
                invoice[header] = content
                continue
        # elif header != '':
        #     if header in constant.EXPORTER_HEADERS:
        #         exporter_header = box
        #         continue
        #     elif header in constant.CONSIGNEE_HEADERS:
        #         consignee_header = box
        #         continue

        # same y and h
        if abs(box[1] - next_box[1]) < 5 and abs(box[3] - next_box[3]) < 5:
            if len(current_row) == 0:
                current_row.append(box)
                current_row_text.append(box_text)
            current_row.append(next_box)
        else:
            # print(current_row)
            if len(current_row) > 2:
                if header_row_found == False:
                    header_row_found, header_texts = is_row_item_header(current_row, img)
                    if header_row_found:
                        header_row = current_row.copy()
                elif header_row_found and is_row_belong_to_header(current_row, header_row):
                    item_rows.append(current_row)
                    items_row_found = True
            else:
                if header_row_found and is_box_belong_to_header(box, header_row):
                    item_rows.append([box])
                    items_row_found = True
            current_row = []

    if header_row_found and items_row_found:
        items = {}
        if len(item_rows) == 1:
            item_rows = split_row(item_rows[0], img)
        print(item_rows)

        for idx, item_row in enumerate(item_rows):
            item_detail = {}
            if len(item_row) == 1:
                item_row = split_col(item_row[0], header_row)
            print(f"{item_row}")
            for i in range(len(header_row)):
                header_text = get_text_from_box(img, header_row[i])
                item_text = get_text_from_box(img, item_row[i])
                item_detail[get_header_text(header_text)] = item_text
                items[f"item_{idx}"] = item_detail

        invoice["ITEMS"] = items
        return invoice

    return invoice