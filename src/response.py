RESPONSES = {

    "greeting": [
        "Hello! Welcome to Wentworth Academy. How can I help you?",
        "Hi! Welcome to Wentworth Academy. What can I help you with?",
        "Hello! What would you like to know about the academy?"
    ],

    "school_location": [
        "The library is located in the Main Academic Building, on the first floor.",
        "The Science Building is next to the Main Academic Building.",
        "The main academic offices are located in the Main Academic Building."
    ],

    "class_schedule": [
        "Classes begin at 8:00 AM. You can check your complete schedule through the student portal.",
        "Your class schedule is available through the student portal.",
        "The academy's regular classes run from 8:00 AM to 3:00 PM."
    ],

    "teacher_information": [
        "Teacher information is available through the faculty directory."
    ],

    "exam_schedule": [
        "You can check the student portal for the complete examination schedule."
    ],

    "club_information": [
        "Wentworth Academy has science, debate, music, art, and sports clubs.",
        "You can join a student club by contacting the club coordinator.",
        "Popular clubs include the Science Club, Debate Club, Music Club, and Art Club."
    ],

    "school_rules": [
        "Students are expected to attend classes regularly and follow the academy's code of conduct.",
        "Students should follow the rules in the Wentworth Academy student handbook.",
        "Students must carry their student ID while on academy grounds."
    ],

    "academic_help": [
        "Sure! Tell me which subject or topic you need help with.",
        "I'd be happy to help. What subject are you working on?",
        "Tell me what you're having difficulty with, and we'll start from there."
    ],

    "goodbye": [
        "Goodbye! See you at Wentworth Academy.",
        "Take care! See you next time.",
        "Goodbye! Have a great day."
    ],

    "unknown": [
        "I'm sorry, I can only help with questions about Wentworth Academy.",
        "I don't have information about that. I can help with academy-related questions.",
        "I'm not sure about that. Try asking me about classes, teachers, exams, clubs, or school rules."
    ]
}


def get_response(intent, message):

    message = message.lower()

    # Teacher information
    if intent == "teacher_information":

        if "chemistry" in message:
            return "Ms. Emily Carter teaches Chemistry."

        if "math" in message or "mathematics" in message:
            return "Mr. Daniel Brooks teaches Mathematics."

        if "history" in message:
            return "Ms. Sophia Bennett teaches History."

        if "biology" in message:
            return "Mr. James Wilson teaches Biology."

        return RESPONSES["teacher_information"][0]


    # School locations
    if intent == "school_location":

        if "library" in message:
            return (
                "The library is located in the Main Academic Building, "
                "on the first floor."
            )

        if "science" in message:
            return (
                "The Science Building is next to the "
                "Main Academic Building."
            )

        return RESPONSES["school_location"][2]


    # Exam information
    if intent == "exam_schedule":

        if "midterm" in message:
            return "Midterm exams are held during the second week of October."

        if "final" in message:
            return "Final exams are held during the first week of December."

        return RESPONSES["exam_schedule"][0]


    # Club information
    if intent == "club_information":

        if "join" in message:
            return (
                "You can join a student club by contacting "
                "the club coordinator."
            )

        if "sport" in message:
            return "Wentworth Academy has several sports clubs."

        return RESPONSES["club_information"][0]


    # Default response
    responses = RESPONSES.get(
        intent,
        RESPONSES["unknown"]
    )

    return responses[0]