def get_line_equation_output_y(eq, x):
    """
    Calculate the y-coordinate value on a line given its equation and x-coordinate.

    Args:
        eq (list): List containing coefficients of the line equation [a, b, c, d].
        x (float): The x-coordinate.

    Returns:
        float: The calculated y-coordinate value.
    """
    return eq[2] * x + eq[3]


def determine_lane(x, y, left_lines_equation, right_lines_equation):
    """
    Determine which lane a point (x, y) belongs to based on given left and right lane lines.

    Args:
        x (int): The x-coordinate of the point.
        y (int): The y-coordinate of the point.
        left_lines_equation (list): List of equations for left lane lines.
        right_lines_equation (list): List of equations for right lane lines.

    Returns:
        str: A string indicating the lane where the point (x, y) belongs ("Middle lane", "Left lane", "Right lane",
             "Other lanes on the left", "Other lanes on the right", or "Unavailable yet").
    """
    if len(right_lines_equation) >= 2 and len(left_lines_equation) >= 2:
        if (y > get_line_equation_output_y(right_lines_equation[0], x)) and \
           (y > get_line_equation_output_y(left_lines_equation[-1], x)):
            return "Middle lane"
        elif (y <= get_line_equation_output_y(left_lines_equation[-1], x)) and \
             (y >= get_line_equation_output_y(left_lines_equation[-2], x)):
            return "Left lane"
        elif y < get_line_equation_output_y(left_lines_equation[-2], x):
            return "Other lanes on the left"
        elif (y <= get_line_equation_output_y(right_lines_equation[0], x)) and \
             (y >= get_line_equation_output_y(right_lines_equation[1], x)):
            return "Right lane"
        elif y < get_line_equation_output_y(right_lines_equation[1], x):
            return "Other lanes on the right"
    else:
        return "Unavailable yet"


def car_in_which_lane(target_car, lanes):
    """
    Determine in which lane a target car is located based on the position of the car and lane lines.

    Args:
        target_car (dict): Dictionary containing information about the target car, including its bounding box coordinates.
        lanes (list): List of lane lines represented as arrays of coordinates.

    Returns:
        str: A string indicating the lane in which the target car is detected ("Middle lane", "Left lane", "Right lane",
             "Other lanes on the left", "Other lanes on the right", or "Unavailable yet").
    """
    left_lines_equation = []
    right_lines_equation = []

    # Extract equations for left and right lane lines
    for line in lanes:
        x1, y1, x2, y2 = line[0]
        equation_para_a = (y2 - y1) / (x2 - x1)
        equation_para_b = y1 - equation_para_a * x1
        line_mid_x = (x1 + x2) / 2
        line_mid_y = (y1 + y2) / 2
        if equation_para_a < 0:
            left_lines_equation.append([line_mid_x, line_mid_y, equation_para_a, equation_para_b])
        elif equation_para_a >= 0:
            right_lines_equation.append([line_mid_x, line_mid_y, equation_para_a, equation_para_b])

    # Sort equations based on slope (a) in descending order
    left_lines_equation = sorted(left_lines_equation, key=lambda x: x[2], reverse=True)
    right_lines_equation = sorted(right_lines_equation, key=lambda x: x[2], reverse=True)

    # Determine which lane the car is in based on its bounding box coordinates
    x1, y1, x2, y2 = target_car['bbox']
    car_in_which_lane_detection_result = determine_lane(int((x1 + x2) / 2), max(y1, y2), left_lines_equation, right_lines_equation)

    return car_in_which_lane_detection_result
