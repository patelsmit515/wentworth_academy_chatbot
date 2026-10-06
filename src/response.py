RESPONSES = {

    "greeting": [
        "Hello! Welcome to Wentworth Academy. How can I help you?",
        "Hi! Welcome to Wentworth Academy. What can I help you with?",
        "Hello! What would you like to know about the academy?"
    ],

    "school_location": [
        "The Main Academic Building is the central building on campus.",
        "The Main Academic Building contains most classrooms and administrative offices.",
        "You can find campus locations through the student portal."
    ],

    "class_schedule": [
        "Classes begin at 8:00 AM. You can check your complete schedule through the student portal.",
        "Your class schedule is available through the student portal.",
        "The academy's regular classes run from 8:00 AM to 3:00 PM."
    ],

    "teacher_information": [
        "You can find teacher information through the faculty directory."
    ],

    "exam_schedule": [
        "You can check the student portal for the complete examination schedule."
    ],

    "club_information": [
        "Wentworth Academy has science, debate, music, art, and sports clubs.",
        "Students can join several academic, creative, and sports clubs.",
        "You can contact the club coordinator if you want to join a student club."
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

        if "physics" in message:
            return "Mr. Michael Anderson teaches Physics."

        if "english" in message:
            return "Ms. Olivia Parker teaches English."

        return RESPONSES["teacher_information"][0]


    # School locations
    if intent == "school_location":

        if "library" in message:
            return (
                "The library is located in the Main Academic Building, "
                "on the first floor. It is open from 8:00 AM to 5:00 PM."
            )

        if "science" in message:
            return (
                "The Science Building is next to the Main Academic Building. "
                "It contains the science laboratories and science classrooms."
            )

        if "laboratory" in message or "lab" in message:
            return (
                "The science laboratories are located in the Science Building, "
                "next to the Main Academic Building."
            )

        if "cafeteria" in message or "canteen" in message:
            return (
                "The cafeteria is located near the Student Center and serves "
                "students during the main school breaks."
            )

        if "student center" in message:
            return (
                "The Student Center is located near the cafeteria and is "
                "available to students during school hours."
            )

        if "office" in message or "administration" in message:
            return (
                "The main administrative offices are located on the ground "
                "floor of the Main Academic Building."
            )

        if "gym" in message or "sports" in message:
            return (
                "The gymnasium is located on the eastern side of campus, "
                "near the sports field."
            )

        return RESPONSES["school_location"][0]


    # Class schedule
    if intent == "class_schedule":

        if "start" in message or "begin" in message:
            return "Classes begin at 8:00 AM on regular school days."

        if "finish" in message or "end" in message:
            return "Regular classes finish at 3:00 PM."

        if "schedule" in message:
            return (
                "You can view your complete class schedule through the "
                "student portal."
            )

        if "monday" in message:
            return (
                "Monday classes follow the regular school timetable. "
                "You can check your student portal for your exact classes."
            )

        return RESPONSES["class_schedule"][0]


    # Exam information
    if intent == "exam_schedule":

        if "midterm" in message:
            return (
                "Midterm exams are held during the second week of October. "
                "Check the student portal for your subject-specific schedule."
            )

        if "final" in message:
            return (
                "Final exams are held during the first week of December. "
                "Check the student portal for your subject-specific schedule."
            )

        if "math" in message or "mathematics" in message:
            return (
                "The Mathematics examination schedule is available through "
                "the student portal."
            )

        if "science" in message or "chemistry" in message:
            return (
                "Science examination dates are listed in the student portal "
                "along with the other subject schedules."
            )

        return RESPONSES["exam_schedule"][0]


    # Club information
    if intent == "club_information":

        if "join" in message:
            return (
                "You can join a student club by contacting the club coordinator "
                "or visiting the Student Center."
            )

        if "sport" in message:
            return (
                "Wentworth Academy has several sports clubs, including "
                "football, basketball, and athletics."
            )

        if "science" in message:
            return (
                "The Science Club organizes experiments, science discussions, "
                "and academic activities for students."
            )

        if "debate" in message:
            return (
                "The Debate Club helps students develop public speaking, "
                "argumentation, and discussion skills."
            )

        if "music" in message:
            return (
                "The Music Club is open to students interested in singing, "
                "instruments, and school performances."
            )

        if "art" in message:
            return (
                "The Art Club organizes creative activities, exhibitions, "
                "and student art projects."
            )

        return RESPONSES["club_information"][0]


    # School rules
    if intent == "school_rules":

        if "id" in message or "student id" in message:
            return (
                "Students must carry their Wentworth Academy student ID "
                "while on academy grounds."
            )

        if "attendance" in message or "absent" in message:
            return (
                "Students are expected to attend classes regularly. "
                "If you need to be absent, follow the academy's absence procedure."
            )

        if "uniform" in message or "dress" in message:
            return (
                "Students are expected to follow the academy's dress code "
                "and uniform guidelines."
            )

        if "phone" in message or "mobile" in message:
            return (
                "Mobile phones should be kept away during classes unless "
                "a teacher allows them for academic activities."
            )

        if "leave" in message:
            return (
                "Students should follow the academy's permission procedure "
                "before leaving campus during school hours."
            )

        return RESPONSES["school_rules"][0]


    # Academic help
    if intent == "academic_help":

        if "math" in message or "mathematics" in message:
            return (
                "I can help with Mathematics topics such as algebra, equations, "
                "geometry, and basic calculations. Tell me what you're working on."
            )

        if "chemistry" in message:
            return (
                "I can help you understand Chemistry topics and work through "
                "problems step by step. What topic are you studying?"
            )

        if "physics" in message:
            return (
                "I can help explain Physics concepts and work through problems "
                "step by step. What topic are you studying?"
            )

        if "biology" in message:
            return (
                "I can help explain Biology concepts and terminology. "
                "What topic are you studying?"
            )

        return RESPONSES["academic_help"][0]


    # Default response
    responses = RESPONSES.get(
        intent,
        RESPONSES["unknown"]
    )

    return responses[0]