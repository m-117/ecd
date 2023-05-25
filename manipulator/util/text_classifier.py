import numpy as np

from transformers import pipeline

def get_emotions(text, model_path):
    classifier = pipeline("text-classification", model=model_path, return_all_scores=True)
    return classifier(text)

def fix_label(label):
    if label == "anger":
        label = "angry"
    if label == "disgust":
        label = "disgusted"
    if label == "joy":
        label = "happy"
    if label == "surprise":
        label = "surprised"
    if label == "sadness":
        label = "sad"
    return label

def analyze_text(text, frames, model_path):
    sentiment_info = []
    full_text_emo = get_emotions(text, model_path)

    multiplier = 1.0
    
    ft_max_score = 0.0
    ft_second_score = 0.0
    ft_max_label = ''
    ft_second_label = ''
    for label in full_text_emo[0]:
        score = label["score"]
        if score > ft_max_score:
            ft_second_score = ft_max_score
            ft_max_score = score
            ft_second_label = ft_max_label
            ft_max_label = label['label']
    
    multiplier = 1.0 + (ft_max_score - ft_second_score)*0.05

    if ft_max_label == "neutral":
        ft_max_label = ft_second_label
        multiplier = 1.0 - (ft_max_score - ft_second_score)*0.1

    ft_max_label = fix_label(ft_max_label)

    sentiment_info.append({"duration": frames,
                            "emotion": ft_max_label,
                            "style": np.zeros(16),
                            "multiplier": multiplier})
    
    # TODO: Implement chunking algorithm for larger text inputs. Not needed for test dataset, but inputs longer than 3 sentences can be emotionally ambiguous.
    # Tokenize input; Calculate sentence/chunk duration (use wav file or create speech); Analyze sentences/chunks and locate emotion switches;    

    return sentiment_info

    

    




