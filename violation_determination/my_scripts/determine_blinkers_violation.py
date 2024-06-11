def determine_blinkers_violation(blinkers_history_timeline, lane_history_timeline):
    dictionary = {"Other lanes on the left": 1, "Left lane": 2, "Middle lane": 3, "Right lane": 4, "Other lanes on the right": 5}
    lane_before = dictionary[lane_history_timeline[0]]
    lane_after = dictionary[lane_history_timeline[-1]]
    print("-----------")
    if lane_after - lane_before > 0:    # blinkers: right
        for blinker in blinkers_history_timeline:
            if blinker == "right":
                return "Right blinker activated => No violation."
    elif lane_after - lane_before < 0:    # blinkers: left
        for blinker in blinkers_history_timeline:
            if blinker == "left":
                return "Left blinker activated => No violation."
    return "Blinkers OFF => Violation detected!"
