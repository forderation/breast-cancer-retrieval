import json
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == "__main__":
    with open('training_binary_sample_400.json') as f:
        training_400_binary = json.load(f)

    with open('training_1_400_subclass.json') as f:
        training_400_subclass = json.load(f)

    ax = plt.figure(figsize=(20, 20))
    sns.set(font_scale=2)
    sns.lineplot(range(0, len(training_400_binary['loss'])), training_400_binary['loss'], linestyle='-', linewidth=3,
                 label="400X BINARY")
    g = sns.lineplot(range(0, len(training_400_subclass['loss'])), training_400_subclass['loss'], linestyle='-',
                     linewidth=3,
                     label="400X SUBCLASS")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.ylim(0, 0.02)
    plt.title('Loss values on main classes')
    plt.savefig('loss-comparing.svg', format='svg')
