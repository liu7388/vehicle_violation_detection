import numpy as np


def merge_similar_lane_detections(lines):
    """
    Merges detected lane lines with similar angles into longer lines.

    Args:
        lines (dict): Dictionary containing detected lane lines grouped by angles.

    Returns:
        list: List of merged lane lines represented as endpoints [[x1, y1, x2, y2]].
    """
    # Initial lists
    numbers = list(lines.keys())

    # Result list
    groups = []
    used_numbers = set()  # Used to mark numbers that have already been grouped

    # Iterate through each number in the initial list
    for i in range(len(numbers)):
        if numbers[i] in used_numbers:
            continue  # Skip if the current number has already been grouped

        current_group = [numbers[i]]  # Create a new group and add the current number to it
        used_numbers.add(numbers[i])  # Mark the current number as grouped

        # Iterate through each number in the initial list
        for j in range(len(numbers)):
            if i != j and numbers[j] not in used_numbers:
                # If another number differs from the current number by <= 3, add it to the current group
                similar_range = 6
                if abs(numbers[i] - numbers[j]) <= similar_range:
                    current_group.append(numbers[j])
                    used_numbers.add(numbers[j])  # Mark the number as grouped

        # Add the current group to the result list
        groups.append(current_group)

    best_lines_merged_list = []

    # Find the longest line in each group and add it to the merged list if length > 50
    for group in groups:
        max_length = 0.0
        best_line_in_group = None
        if len(group) > 1:
            for theda in group:
                x1, y1, x2, y2 = lines[theda][0]
                length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                if length > max_length:
                    max_length = length
                    best_line_in_group = lines[theda]
        else:
            theda = group[0]
            x1, y1, x2, y2 = lines[theda][0]
            max_length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            best_line_in_group = lines[theda]

        if max_length > 50:
            best_lines_merged_list.append(best_line_in_group)

    print("Lanes detected:", len(groups))
    print("Theda groups:", groups, end="\n\n")

    return best_lines_merged_list
