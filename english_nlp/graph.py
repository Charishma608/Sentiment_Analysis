import pandas as pd
import matplotlib.pyplot as plt

emotion_mapping = {
    "0": "sad",
    "1": "happy",
    "2": "love",
    "3": "anger",
    "4": "fear",
    "5": "surprise",
}

df = pd.read_csv("Dataset.csv")

emotion_counts = df["emotion"].value_counts().sort_index()

emotion_labels = [emotion_mapping[str(i)] for i in emotion_counts.index]

plt.figure(figsize=(12, 6))
bars = plt.bar(emotion_labels, emotion_counts.values)

plt.title("Distribution of Emotions in Sentences", fontsize=16)
plt.xlabel("Emotions", fontsize=12)
plt.ylabel("Number of Sentences", fontsize=12)

for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2.0,
        height,
        f"{height}",
        ha="center",
        va="bottom",
    )

colors = ["blue", "green", "pink", "red", "purple", "orange"]
for bar, color in zip(bars, colors):
    bar.set_color(color)

plt.legend(bars, emotion_labels)

plt.tight_layout()
plt.show()

print("Emotion counts:")
for emotion, count in zip(emotion_labels, emotion_counts.values):
    print(f"{emotion}: {count}")
