import cv2 as cv

def draw_bboxes(image, pred_df):
  for i, det in enumerate(pred_df.to_numpy()):
    xmin, ymin, xmax, ymax = int(det[0]), int(det[1]), int(det[2]), int(det[3])
    conf = det[4]
    text = f'{i}) conf: {conf:.2f}'

    # Draw bounding box
    bbox_color = (0,0,255)
    image = cv.rectangle(image, (xmin,ymin), (xmax,ymax), bbox_color, 2)
    # Add text
    font_size = 1
    font_thickness = 3
    text_color = (255,255,255)
    text_size, _ = cv.getTextSize(text, cv.FONT_HERSHEY_SIMPLEX, font_size, font_thickness)
    text_w, text_h = text_size
    image = cv.rectangle(image, (xmin, ymin), (xmin + text_w, ymin + text_h), bbox_color, -1)
    image = cv.putText(image, text, (xmin, ymin + text_h + font_size - 1), cv.FONT_HERSHEY_SIMPLEX, font_size, text_color, font_thickness)

  return image