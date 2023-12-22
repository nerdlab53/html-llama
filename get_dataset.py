from datasets import load_dataset
def get_dataset(dataset_name : str):
    if dataset_name == "":
        dataset = load_dataset("retr0sushi04/html", split = "train")
        dataset = dataset.train_test_split(test_size=0.2)
    else:
        dataset = load_dataset(dataset_name, split = "train")
        dataset = dataset.train_test_split(test_size=0.2)
    return dataset