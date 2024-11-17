import json
from datetime import datetime

import code_utils
from code_utils.utils import save_annotation, get_now
import numpy as np
import matplotlib.patheffects as PathEffects
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.manifold import TSNE
from pathlib import Path
from code_utils import open_smell_file
from sklearn.preprocessing import LabelEncoder


class TSNECreator:
    @staticmethod
    def _scatter(x, label_codes, labels_unique, subtitle=None):
        # We choose a color palette with seaborn.
        palette = np.array(sns.color_palette("hls", 10))

        plt.tight_layout()
        # We create a scatter plot.
        f = plt.figure(figsize=(8, 8))

        ax = f.add_subplot(aspect="equal")

        sc = ax.scatter(x[:, 0], x[:, 1], lw=0, s=40,
                        c=palette[label_codes.astype(int)])

        plt.xlim(-25, 25)
        plt.ylim(-25, 25)
        ax.axis('off')
        ax.axis('tight')

        # We add the labels for each digit.
        txts = []
        for i in range(len(labels_unique)):
            # Position of each label.
            xtext, ytext = np.median(x[label_codes == i, :], axis=0)
            txt = ax.text(xtext, ytext, labels_unique[i], fontsize=24)
            txt.set_path_effects([
                PathEffects.Stroke(linewidth=5, foreground="w"),
                PathEffects.Normal()])
            txts.append(txt)

        now = get_now()

        plt.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)

        save_path = "/home/eislamoglu/Pictures/tsne/tsne_" + now + "_" + subtitle + ".png"
        plt.savefig(save_path, dpi=500, bbox_inches=0)

        return save_path, f

    @staticmethod
    def load_embeds(data_path, label_path):
        data_path = Path(data_path)

        data = np.load(data_path, "r")
        data = data.reshape(-1, np.prod(data.shape[1:]))  # flatten

        label_df = open_smell_file(label_path)

        return data, label_df

    @staticmethod
    def tsne(data, label_df):
        labels = label_df.smellKey

        le = LabelEncoder()
        label_codes = le.fit_transform(labels)
        labels_unique = le.classes_

        tsne = TSNE()
        print(">>> Start TSNE fit")
        tsne_embeds = tsne.fit_transform(data)
        print(">>> End TSNE fit")

        return tsne_embeds, label_codes, labels_unique

    @staticmethod
    def create_single(data_path, label_path, display=False):
        data_path = Path(data_path)

        data, label_df = TSNECreator.load_embeds(data_path, label_path)

        tsne_embeds, label_codes, labels_unique = TSNECreator.tsne(data, label_df)

        _, f = TSNECreator._scatter(tsne_embeds, label_codes, labels_unique, data_path.stem)
        save_annotation("tsne_" + data_path.stem, "Created tsne. Class codes for values are like" + str(labels_unique))

        if display:
            f.show(True)

    @staticmethod
    def create_from_pair(original_data_path, new_data_path, label_path, display=False):
        """original_data must have same shape as new data
        """
        original_data_path = Path(original_data_path)
        new_data_path = Path(new_data_path)

        original_data, label_df = TSNECreator.load_embeds(original_data_path, label_path)
        new_data, label_df = TSNECreator.load_embeds(new_data_path, label_path)

        assert original_data.shape == new_data.shape

        eval_original_tsne_embeds, label_codes, labels_unique = TSNECreator.tsne(original_data, label_df)
        eval_new_tsne_embeds, _, _ = TSNECreator.tsne(new_data, label_df)

        _, f_original = TSNECreator._scatter(eval_original_tsne_embeds, label_codes, labels_unique, original_data_path.stem)
        _, f_new = TSNECreator._scatter(eval_new_tsne_embeds, label_codes, labels_unique, new_data_path.stem)
        if display:
            f_original.show(True)
            f_new.show(True)

        save_annotation("tsne_" + original_data_path.stem, "Created tsne. Class codes for values are like" + str(labels_unique))


if __name__ == '__main__':
    TSNECreator.create_from_pair(
        "/home/user/PycharmProjects/Model_Scratch/data/1500_smells_graphcodebert_pooler_output.npy",
        "/home/user/Desktop/Triplet-net-keras/Test/embeds_nli_pooler_1500_1.npy",
        "/home/user/PycharmProjects/Model_Scratch/data/raw/7500_smells_test.json",
        True
    )