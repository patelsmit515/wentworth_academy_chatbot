from src.response import get_response


intents = [
    "greeting",
    "school_location",
    "class_schedule",
    "teacher_information",
    "exam_schedule",
    "club_information",
    "school_rules",
    "academic_help",
    "goodbye",
    "unknown"
]


print("\nResponse Layer Test")
print("-------------------")

for intent in intents:

    response = get_response(intent)

    print(f"{intent} -> {response}")
    
    