import numpy as np


def merge_similar_lane_detections(lines):
    # 初始列表
    numbers = list(lines.keys())

    # 結果列表
    groups = []
    used_numbers = set()  # 用於標記已經分組的數字

    # 遍歷初始列表中的每個數字
    for i in range(len(numbers)):
        if numbers[i] in used_numbers:
            continue  # 如果當前數字已經被分組，跳過

        current_group = [numbers[i]]  # 創建一個新分組，並將當前數字添加到分組中
        used_numbers.add(numbers[i])  # 標記當前數字已經被分組

        # 遍歷初始列表中的每個數字
        for j in range(len(numbers)):
            if i != j and numbers[j] not in used_numbers:
                # 如果另一個數字與當前數字的差的絕對值小於等於3，則將該數字添加到當前分組中
                similar_range = 6
                if abs(numbers[i] - numbers[j]) <= similar_range:
                    current_group.append(numbers[j])
                    used_numbers.add(numbers[j])  # 標記該數字已經被分組

        # 將當前分組添加到結果列表中
        groups.append(current_group)

    best_lines_merged_list = []

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
        # 硬條件：無論如何長度都要超過50
        if max_length > 50:
            best_lines_merged_list.append(best_line_in_group)

    # print("best_lines_merged_list:", best_lines_merged_list)

    print("Lanes detected:", len(groups))
    print("Theda groups:", groups, end="\n\n")
    # if len(groups) >= 4:
    #     exit()

    return best_lines_merged_list
