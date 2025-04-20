import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
"""
To do: save it as jpg. Include more analysis. try to create a report for all the analysis done (in pdf).

"""
class DataAnalyzer:

    def __init__(self, data_path = "data/preprocessed_data.csv"):
        self.df = pd.read_csv(data_path)
        

    def show_class_distribution(self):
        self.df["label"].value_counts().plot(kind="bar", title="Class Distribution")
        plt.xlabel("Label")
        plt.ylabel("Count")
        plt.show()

    def show_correlation_matrix(self):
        sns.heatmap(self.df.corr(), annot=True, cmap="coolwarm")
        plt.title("Corrlation Matrix")
        plt.show()


if __name__ == "__main__":
    analyzer = DataAnalyzer()
    analyzer.show_class_distribution()
    analyzer.show_correlation_matrix()

