def get_line_equation_output_y(eq, x):
    return eq[2] * x + eq[3]


def determine_lane(x, y, left_lines_equation, right_lines_equation):
    if len(right_lines_equation) >= 2 and len(left_lines_equation) >= 2:
        if ((y > get_line_equation_output_y(right_lines_equation[0], x)) and
                (y > get_line_equation_output_y(left_lines_equation[-1], x))):
            return "Middle lane"
        elif ((y <= get_line_equation_output_y(left_lines_equation[-1], x)) and
                (y >= get_line_equation_output_y(left_lines_equation[-2], x))):
            return "Left lane"
        elif y < get_line_equation_output_y(left_lines_equation[-2], x):
            return "Other lanes on the left"
        elif ((y <= get_line_equation_output_y(right_lines_equation[0], x)) and
                (y >= get_line_equation_output_y(right_lines_equation[1], x))):
            return "Right lane"
        elif y < get_line_equation_output_y(right_lines_equation[1], x):
            return "Other lanes on the right"
    else:
        # Number of lanes detected either on the left or right is less than 2 !
        return "Unavailable yet"


def car_in_which_lane(target_car, lanes):
    left_lines_equation = []
    right_lines_equation = []
    for line in lanes:
        x1, y1, x2, y2 = line[0]
        equation_para_a = (y2 - y1) / (x2 - x1)
        equation_para_b = y1 - equation_para_a * x1
        line_mid_x = (x1 + x2) / 2
        line_mid_y = (y1 + y2) / 2
        if equation_para_a < 0:
            left_lines_equation.append(list([line_mid_x, line_mid_y, equation_para_a, equation_para_b]))
        elif equation_para_a >= 0:
            right_lines_equation.append(list([line_mid_x, line_mid_y, equation_para_a, equation_para_b]))
    left_lines_equation = sorted(left_lines_equation, key=lambda x: x[2], reverse=True)
    right_lines_equation = sorted(right_lines_equation, key=lambda x: x[2], reverse=True)

    # print(left_lines_equation, right_lines_equation)

    x1, y1, x2, y2 = target_car['bbox']
    car_in_which_lane_detection_result = (determine_lane(int((x1 + x2) / 2), max(list([y1, y2])), left_lines_equation, right_lines_equation))

    return car_in_which_lane_detection_result
