def analyze_mental_state(emotion, depression_score):
    """
    Combine emotion and depression prediction
    to determine deeper mental state
    """

    if depression_score == 1:

        if emotion in ["sad", "fear", "neutral"]:
            return "High Depression Risk"

        elif emotion == "angry":
            return "Emotional Instability"

        else:
            return "Masked Depression"

    if depression_score == 0:

        if emotion == "happy":
            return "Emotionally Stable"

        elif emotion == "neutral":
            return "Mild Stress"

        elif emotion == "angry":
            return "High Stress"

        elif emotion == "fear":
            return "Anxiety Signs"

        elif emotion == "sad":
            return "Low Mood"

        else:
            return "Normal Emotional Variation"

    return "Unknown State"