RESPONSES = {

    "greeting":
        "Hello! Welcome to Wentworth Academy. How can I help you?",

    "school_location":
        "The library is located in the Main Academic Building.",

    "class_schedule":
        "You can check your class schedule through the student portal.",

    "teacher_information":
        "Teacher information is available through the faculty directory.",

    "exam_schedule":
        "You can find upcoming exams on the examination schedule.",

    "club_information":
        "Wentworth Academy offers several student clubs and activities.",

    "school_rules":
        "Please check the student handbook for the complete school rules.",

    "academic_help":
        "Sure! Tell me which subject or topic you need help with.",

    "goodbye":
        "Goodbye! See you at Wentworth Academy.",

    "unknown":
        "I'm sorry, I don't have information about that."
}


def get_response(intent):

    return RESPONSES.get(
        intent,
        RESPONSES["unknown"]
    )