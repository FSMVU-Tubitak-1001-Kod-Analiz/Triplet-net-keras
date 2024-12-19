from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context

import sys
sys.path.append("/home/user/Desktop/Triplet-net-keras/triplet_net/triplet_net")
from triplet_net.triplet_create import create_triplet
import itertools
import os
from pathlib import Path


def _pool(embeds_path, train_label_path, test_label_path, save_path, train_batch_size, epoch):
    print("Starting run with parameters", embeds_path, train_label_path, test_label_path, save_path, train_batch_size, epoch)
    create_triplet(embeds_path, train_label_path, test_label_path, save_path, train_batch_size, epoch)


if __name__ == "__main__":
    train_paths = [
        # "/home/user/PycharmProjects/Model_Scratch/data/raw/9000_smells_train.json",
        # "/home/user/PycharmProjects/Model_Scratch/data/raw/9000_smells_train_java.json",
        "/home/user/PycharmProjects/Model_Scratch/data/raw/9000_smells_train_python.json",
        # "/home/user/PycharmProjects/Model_Scratch/data/raw/9000_smells_train_php.json"
    ]

    test_paths = [
        # "/home/user/PycharmProjects/Model_Scratch/data/raw/9000_smells_test.json",
        # "/home/user/PycharmProjects/Model_Scratch/data/raw/9000_smells_test_java.json",
        "/home/user/PycharmProjects/Model_Scratch/data/raw/9000_smells_test_python.json",
        # "/home/user/PycharmProjects/Model_Scratch/data/raw/9000_smells_test_php.json"
    ]

    embeds_paths = [
        "/home/user/PycharmProjects/Model_Scratch/data/9000_smells_codebert_pooler_output.npy",
        "/home/user/PycharmProjects/Model_Scratch/data/9000_smells_bert_nli_mean_token_pooler_output.npy",
        # "/home/user/PycharmProjects/Model_Scratch/data/9000_smells_graphcodebert_hidden_state.npy",
        "/home/user/PycharmProjects/Model_Scratch/data/9000_smells_graphcodebert_pooler_output.npy"
    ]

    train_embeds_paths = [
        "/home/user/PycharmProjects/Model_Scratch/data/9000_smells_codebert_pooler_output_train.npy",
        "/home/user/PycharmProjects/Model_Scratch/data/9000_smells_bert_nli_mean_token_pooler_output_train.npy",
        # "/home/user/PycharmProjects/Model_Scratch/data/9000_smells_graphcodebert_hidden_state_train.npy",
        "/home/user/PycharmProjects/Model_Scratch/data/9000_smells_graphcodebert_pooler_output_train.npy"
    ]

    test_embeds_paths = [
        "/home/user/PycharmProjects/Model_Scratch/data/9000_smells_codebert_pooler_output_test.npy",
        "/home/user/PycharmProjects/Model_Scratch/data/9000_smells_bert_nli_mean_token_pooler_output_test.npy",
        # "/home/user/PycharmProjects/Model_Scratch/data/9000_smells_graphcodebert_hidden_state_test.npy",
        "/home/user/PycharmProjects/Model_Scratch/data/9000_smells_graphcodebert_pooler_output_test.npy"
    ]

    with ProcessPoolExecutor(max_workers=9) as executor:
        futures = []

        for (train_label_path, test_label_path), embeds_path in itertools.product(zip(train_paths, test_paths), embeds_paths):
            lang = Path(train_label_path).stem.split("_")[-1]
            model_name = "_".join(Path(embeds_path).stem.split("_")[2:])
            save_folder = "Test/revision2"

            save_path = os.path.join(save_folder, lang + "_" + model_name + ".npy")

            futures.append(executor.submit(_pool, embeds_path, train_label_path, test_label_path, save_path, 1024, 30))

        for i in as_completed(futures):
            print(f'About {len(executor._pending_work_items)} tasks remain')
            print(i.result())

    # for (train_label_path, test_label_path), embeds_path in itertools.product(
    #         zip(train_paths, test_paths),
    #         ["/home/user/PycharmProjects/Model_Scratch/data/7500_smells_graphcodebert_hidden_state.npy"]):
    #     lang = Path(train_label_path).stem.split("_")[-1]
    #     model_name = "_".join(Path(embeds_path).stem.split("_")[2:])
    #     save_folder = "Test/revision"
    #
    #     save_path = os.path.join(save_folder, lang + "_" + model_name + ".npy")
    #
    #     _pool(embeds_path, train_label_path, test_label_path, save_path, 1024, 30)
    # running_params = [
    #     {
    #         "embeds_path": "/home/user/PycharmProjects/Model_Scratch/data/7500_smells_codebert_pooler_output.npy",
    #         "save_path": "Test/embeds_codebert_pooler_1500_1.npy",
    #     },
    #     {
    #         "embeds_path": "/home/user/PycharmProjects/Model_Scratch/data/7500_smells_bert_nli_mean_token_pooler_output.npy",
    #         "save_path": "Test/embeds_nli_pooler_1500_1.npy"
    #     },
    #     {
    #         "embeds_path": "/home/user/PycharmProjects/Model_Scratch/data/7500_smells_graphcodebert_hidden_state.npy",
    #         "save_path": "Test/embeds_graphcodebert_1500_1.npy"
    #     },
    #     {
    #         "embeds_path": "/home/user/PycharmProjects/Model_Scratch/data/7500_smells_graphcodebert_pooler_output.npy",
    #         "save_path": "Test/embeds_graphcodebert_pooler_1500_1.npy"
    #     }
    # ]
    #
    # for i in running_params:
    #     create_triplet(**i, train_label_path="/home/user/PycharmProjects/Model_Scratch/data/raw/7500_smells_train.json",
    #                    test_label_path="/home/user/PycharmProjects/Model_Scratch/data/raw/7500_smells_test.json",
    #                    train_batch_size=1024, epoch=30)
