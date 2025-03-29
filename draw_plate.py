import cv2
import numpy as np

def overlay_images(background, overlay, position=(0, 0), alpha=0.5):
    """
    Overlay the overlay image on the background image at a specific position with transparency alpha.
    
    Args:
    background (numpy.ndarray): The background image.
    overlay (numpy.ndarray): The overlay image.
    position (tuple): The (x, y) position to place the overlay on the background.
    alpha (float): The transparency of the overlay image (0: completely transparent, 1: completely opaque).
    
    Returns:
    numpy.ndarray: The resulting image after overlay.
    """
    x, y = position
    h, w = overlay.shape[0], overlay.shape[1]
    
    # Ensure overlay image does not exceed background image size
    if x + w > background.shape[1] or y + h > background.shape[0]:
        raise ValueError("Overlay image exceeds background image size at the specified position.")
    
    # Create a region of interest (ROI) on the background image
    roi = background[y:y+h, x:x+w]
    
    # Blend images using alpha transparency
    blended = cv2.addWeighted(roi, 1 - alpha, overlay, alpha, 0)
    
    # Place blended result back into the background image
    background[y:y+h, x:x+w] = blended
    
    return background


def overlay_plates_on_frame(frame, plates_dict, right_padding, y_start_show_plate, y_magin, plate_size=(300, 200), transparency=0.5):
    """
    Overlay resized plates and text onto the frame.

    :param frame: Numpy array, the base frame to overlay on
    :param plates_dict: Dict, key=plate number (str), value=plate image (numpy array)
    :param plate_size: Tuple, size to resize the plate images (width, height)
    """
    frame_height, frame_width, _ = frame.shape
    start_y = frame_height // 4  # Start from 1/3 of the frame height
    x_position = frame_width - plate_size[0] - right_padding  # Align right

    # Iterate through plates in reverse order (last key at the bottom)
    for i, plate_number in enumerate(reversed(plates_dict)):
        # Resize the plate image
        plate_img = plates_dict[plate_number]
        resized_plate = cv2.resize(plate_img, plate_size)

        # Calculate y position for this plate
        y_position = start_y + i * (plate_size[1] + y_magin)

        # Ensure it doesn't go out of the frame
        if y_position + plate_size[1] > frame_height:
            break

        # Overlay the plate on the frame
        frame = overlay_images(frame, resized_plate, (x_position, y_position), alpha=transparency)

        # Add text with a background
        text = plate_number
        font_scale = 1
        thickness = 2
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]

        # Text position: top-left corner of the plate
        text_x = x_position
        text_y = y_position + text_size[1] + 5

        # Draw background for the text
        cv2.rectangle(frame,
                      (text_x, text_y - text_size[1] - 5), 
                      (text_x + text_size[0] + 10, text_y + 5),
                      (0, 165, 255),  # Orange background
                      -1)

        # Put text on the frame
        cv2.putText(frame, 
                    text, 
                    (text_x + 5, text_y), 
                    0, 
                    font_scale, 
                    (255, 255, 255),  # White text
                    thickness,
                    lineType=cv2.LINE_AA)

    return frame

# Example usage:
if __name__ == "__main__":
    # Create a sample frame
    frame = np.zeros((600, 800, 3), dtype=np.uint8)

    # Sample dict of plates (key=plate number, value=plate image)
    plates = {
        "29A-12345": np.random.randint(0, 256, (200, 400, 3), dtype=np.uint8),
        "30B-67890": np.random.randint(0, 256, (200, 400, 3), dtype=np.uint8),
        "77C-11122": np.random.randint(0, 256, (200, 400, 3), dtype=np.uint8)
    }

    # Overlay plates on the frame
    frame_with_plates = overlay_plates_on_frame(frame, plates)

    # Display the result
    cv2.imshow("Frame with Plates", frame_with_plates)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
