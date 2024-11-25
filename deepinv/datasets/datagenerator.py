from tqdm import tqdm
import os
import h5py
import torch
from torch.utils.data import DataLoader, Subset
from torch.utils import data

import numpy as np

class InMemoryDataset(data.Dataset):
    def __init__(self, data_store, train=True, transform=None):
        super().__init__()
        self.transform = transform
        self.unsupervised = False
        if train:
            # self.x = data_store.get("x_train", None)
            self.x = data_store["x_train"]
            self.y = data_store["y_train"]
            self.unsupervised = self.x is None
        else:
            # self.x = data_store.get("x_test", None)
            self.x = data_store["x_train"]
            self.y = data_store["y_test"]

    def __getitem__(self, index):
        y = torch.from_numpy(self.y[index]).float()
        x = y if self.unsupervised else torch.from_numpy(self.x[index]).float()
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.y)


def generate_dataset_in_memory(
    train_dataset,
    physics,
    device="cpu",
    train_datapoints=None,
    test_datapoints=None,
    batch_size=4,
    supervised=True,
    verbose=True,
    show_progress_bar=False,
):
    data = {"x_train": [], "y_train": [], "x_test": [], "y_test": []}

    def process_data(dataloader, key_x, key_y):
        for batch in dataloader:
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            x = x.to(device)
            y = physics(x)
            data[key_y].append(y.cpu().numpy())
            if supervised:
                data[key_x].append(x.cpu().numpy())

    if train_dataset:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            num_workers=0,
            shuffle=True,
        )
        process_data(train_loader, "x_train", "y_train")

    data["x_train"] = None if not supervised else np.concatenate(data["x_train"], axis=0)
    data["y_train"] = np.concatenate(data["y_train"], axis=0)

    return data


class InMemoryDataset1(data.Dataset):
    def __init__(self, data_store, train=True, transform=None):
        super().__init__()
        self.train = train
        self.unsupervised = "x_train" not in data_store
        self.transform = transform
        self.y = data_store["y_train"]
        self.x = data_store["x_train"] if not self.unsupervised else None

    def __getitem__(self, index):
        y = self.y[index].float()

        x = y
        if not self.unsupervised:
            x = self.x[index].float()

        if self.transform:
            x = self.transform(x)

        return x, y

    def __len__(self):
        return len(self.y)


def generate_dataset_in_memory1(
    train_dataset,
    physics,
    device="cpu",
    train_datapoints=None,
    batch_size=4,
    num_workers=0,
    supervised=True,
    verbose=True,
    show_progress_bar=False,
):
    if train_dataset is None:
        raise ValueError("No train dataset provided.")

    if not isinstance(physics, list):
        physics = [physics]
        G = 1
    else:
        G = len(physics)

    if train_datapoints is not None:
        datapoints = int(train_datapoints)
    else:
        datapoints = len(train_dataset)

    n_train = datapoints
    n_train_g = int(n_train / G)
    n_dataset_g = int(len(train_dataset) / G)

    # In-memory storage
    data_store = {
        "y_train": [],
        "x_train": [] if supervised else None
    }

    for g in range(G):
        if train_dataset is not None:
            x = train_dataset[0]
        elif test_dataset is not None:
            x = test_dataset[0]

        x = x[0] if isinstance(x, list) or isinstance(x, tuple) else x
        x = x.to(device).unsqueeze(0)

        epochs = int(n_train_g / len(train_dataset)) + 1

        for e in (
            progress_bar := tqdm(
                range(epochs),
                ncols=150,
                disable=(not verbose or not show_progress_bar),
            )
        ):  
            # print(f"int(n_train_g / len(train_dataset)) + 1: {int(n_train_g / len(train_dataset)) + 1}")
            desc = (
                f"Generating dataset operator {g + 1}"
                if G > 1
                else "Generating train dataset"
            )
            progress_bar.set_description(desc)

            train_dataloader = DataLoader(
                Subset(
                    train_dataset,
                    indices=list(range(g * n_dataset_g, (g + 1) * n_dataset_g)),
                ),
                batch_size=batch_size,
                num_workers=num_workers,
                pin_memory=(device != "cpu"),
            )
            
            
            batches = len(train_dataloader) - int(train_dataloader.drop_last)
            # print(f"batches: {batches}/n len(train dataloader): {len(train_dataloader)}\nint(train_dataloader.drop_last):{int(train_dataloader.drop_last)}")
            iterator = iter(train_dataloader)
            # raise ValueError(f"G: {G}\nepochs:{epochs}\nbatches: {batches}")
            for _ in range(batches):

                x = next(iterator)
                x = x[0] if isinstance(x, (list, tuple)) else x
                x = x.to(device)

                # Generate measurements
                y = physics[g](x)

                # Store data in memory
                data_store["y_train"].append(y.to(device))
                if supervised:
                    data_store["x_train"].append(x.to(device))
                print(f"len(data_store['y_train']):{len(data_store['y_train'])}")
                    

    # Concatenate lists to form tensors
    data_store["y_train"] = torch.cat(data_store["y_train"])
    print(f"len(data_store['y_train']):{len(data_store['y_train'])}")
    if supervised:
        data_store["x_train"] = torch.cat(data_store["x_train"])
    # raise ValueError(f"len(data_store['y_train']):{len(data_store['y_train'])}")

    if verbose:
        print("Dataset has been generated in memory.")

    return data_store


class HDF5Dataset(data.Dataset):
    r"""
    DeepInverse HDF5 dataset with signal/measurement pairs.

    :param str path: Path to the folder containing the dataset (one or multiple HDF5 files).
    :param bool train: Set to ``True`` for training and ``False`` for testing.
    :param transform: A torchvision transform to apply to the data.
    """

    def __init__(self, path, train=True, transform=None):
        super().__init__()
        self.data_info = []
        self.data_cache = {}
        self.unsupervised = False
        self.transform = transform
        hd5 = h5py.File(path, "r")
        if train:
            if "x_train" in hd5:
                self.x = hd5["x_train"]
            else:
                self.unsupervised = True
            self.y = hd5["y_train"]
        else:
            self.x = hd5["x_test"]
            self.y = hd5["y_test"]

    def __getitem__(self, index):
        y = torch.from_numpy(self.y[index]).type(torch.float)

        x = y
        if not self.unsupervised:
            x = torch.from_numpy(self.x[index]).type(torch.float)

        if self.transform is not None:
            x = self.transform(x)

        return x, y

    def __len__(self):
        return len(self.y)


def generate_dataset(
    train_dataset,
    physics,
    save_dir,
    test_dataset=None,
    device="cpu",
    train_datapoints=None,
    test_datapoints=None,
    physics_generator=None,
    dataset_filename="dinv_dataset",
    batch_size=4,
    num_workers=0,
    supervised=True,
    verbose=True,
    show_progress_bar=False,
):
    r"""
    Generates dataset of signal/measurement pairs from base dataset.

    It generates the measurement data using the forward operator provided by the user.
    The dataset is saved in HD5 format and can be easily loaded using the HD5Dataset class.
    The generated dataset contains a train and test splits.

    :param torch.data.Dataset train_dataset: base dataset (e.g., MNIST, CelebA, etc.)
        with images used for generating associated measurements
        via the chosen forward operator. The generated dataset is saved in HD5 format and can be easily loaded using the
        HD5Dataset class.
    :param deepinv.physics.Physics physics: Forward operator used to generate the measurement data.
        It can be either a single operator or a list of forward operators. In the latter case, the dataset will be
        assigned evenly across operators.
    :param str save_dir: folder where the dataset and forward operator will be saved.
    :param torch.data.Dataset test_dataset: if included, the function will also generate measurements associated to the
        test dataset.
    :param torch.device device: which indicates cpu or gpu.
    :param int, None train_datapoints: Desired number of datapoints in the training dataset. If set to ``None``, it will use the
        number of datapoints in the base dataset. This is useful for generating a larger train dataset via data
        augmentation (which should be chosen in the train_dataset).
    :param int, None test_datapoints: Desired number of datapoints in the test dataset. If set to ``None``, it will use the
        number of datapoints in the base test dataset.
    :param None, deepinv.physics.generator.PhysicsGenerator physics_generator: Optional physics generator for generating
            the physics operators. If not None, the physics operators are randomly sampled at each iteration using the generator.
    :param str dataset_filename: desired filename of the dataset.
    :param int batch_size: batch size for generating the measurement data
        (it only affects the speed of the generating process)
    :param int num_workers: number of workers for generating the measurement data
        (it only affects the speed of the generating process)
    :param bool supervised: Generates supervised pairs (x,y) of measurements and signals.
        If set to ``False``, it will generate a training dataset with measurements only (y)
        and a test dataset with pairs (x,y)
    :param bool verbose: Output progress information in the console.
    :param bool show_progress_bar: Show progress bar during the generation
        of the dataset (if verbose is set to True).

    """
    if os.path.exists(os.path.join(save_dir, dataset_filename)):
        print(
            "WARNING: Dataset already exists, this will overwrite the previous dataset."
        )

    if test_dataset is None and train_dataset is None:
        raise ValueError("No train or test datasets provided.")

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if not (type(physics) in [list, tuple]):
        physics = [physics]
        G = 1
    else:
        G = len(physics)

    if train_dataset is not None:
        if train_datapoints is not None:
            datapoints = int(train_datapoints)
        else:
            datapoints = len(train_dataset)

        n_train = datapoints  # min(len(train_dataset), datapoints)
        n_train_g = int(n_train / G)
        n_dataset_g = int(min(len(train_dataset), datapoints) / G)

    if test_dataset is not None:
        test_datapoints = (
            test_datapoints if test_datapoints is not None else len(test_dataset)
        )
        n_test = min(len(test_dataset), test_datapoints)
        n_test_g = int(n_test / G)

    hf_paths = []

    for g in range(G):
        hf_path = f"{save_dir}/{dataset_filename}{g}.h5"
        hf_paths.append(hf_path)
        hf = h5py.File(hf_path, "w")

        hf.attrs["operator"] = physics[g].__class__.__name__

        if train_dataset is not None:
            x = train_dataset[0]
        elif test_dataset is not None:
            x = test_dataset[0]

        x = x[0] if isinstance(x, list) or isinstance(x, tuple) else x
        x = x.to(device).unsqueeze(0)

        # choose operator and generate measurement
        if physics_generator is not None:
            params = physics_generator.step(batch_size=batch_size)
            y = physics[g](x, **params)
        else:
            y = physics[g](x)

        # TODO save params if physics_generator is not None
        torch.save(physics[g].state_dict(), f"{save_dir}/physics{g}.pt")

        if train_dataset is not None:
            hf.create_dataset("y_train", (n_train_g,) + y.shape[1:], dtype="float")
            if supervised:
                hf.create_dataset("x_train", (n_train_g,) + x.shape[1:], dtype="float")

            index = 0

            epochs = int(n_train_g / len(train_dataset)) + 1
            
            # print(f"int(n_train_g / len(train_dataset)) + 1: {int(n_train_g / len(train_dataset)) + 1}")
            for e in (
                progress_bar := tqdm(
                    range(epochs),
                    ncols=150,
                    disable=(not verbose or not show_progress_bar),
                )
            ):
                desc = (
                    f"Generating dataset operator {g + 1}"
                    if G > 1
                    else "Generating train dataset"
                )
                progress_bar.set_description(desc)

                train_dataloader = DataLoader(
                    Subset(
                        train_dataset,
                        indices=list(range(g * n_dataset_g, (g + 1) * n_dataset_g)),
                    ),
                    batch_size=batch_size,
                    num_workers=num_workers,
                    pin_memory=False if device == "cpu" else True,
                )

                batches = len(train_dataloader) - int(train_dataloader.drop_last)
                iterator = iter(train_dataloader)
                    
                # raise ValueError(f"G: {G}\nepochs:{epochs}\nbatches: {batches}")
                for _ in range(batches):
                    x = next(iterator)
                    x = x[0] if isinstance(x, list) or isinstance(x, tuple) else x
                    x = x.to(device)

                    # choose operator and generate measurement
                    y = physics[g](x)

                    # Add new data to it
                    bsize = x.size()[0]

                    if bsize + index > n_train_g:
                        bsize = n_train_g - index

                    hf["y_train"][index : index + bsize] = (
                        y[:bsize, :].to("cpu").numpy()
                    )
                    if supervised:
                        hf["x_train"][index : index + bsize] = (
                            x[:bsize, ...].to("cpu").numpy()
                        )
                    index = index + bsize

        if test_dataset is not None:
            index = 0
            test_dataloader = DataLoader(
                Subset(
                    test_dataset, indices=list(range(g * n_test_g, (g + 1) * n_test_g))
                ),
                batch_size=batch_size,
                num_workers=num_workers,
                pin_memory=True,
            )

            batches = len(test_dataloader) - int(test_dataloader.drop_last)
            iterator = iter(test_dataloader)
            for i in range(batches):
                x = next(iterator)
                x = x[0] if isinstance(x, list) or isinstance(x, tuple) else x
                x = x.to(device)

                # choose operator
                y = physics[g](x)

                if i == 0:  # create dict
                    hf.create_dataset(
                        "x_test", (n_test_g,) + x.shape[1:], dtype="float"
                    )
                    hf.create_dataset(
                        "y_test", (n_test_g,) + y.shape[1:], dtype="float"
                    )

                # Add new data to it
                bsize = x.size()[0]
                hf["x_test"][index : index + bsize] = x.to("cpu").numpy()
                hf["y_test"][index : index + bsize] = y.to("cpu").numpy()
                index = index + bsize
        hf.close()

    if verbose:
        print("Dataset has been saved in " + str(save_dir))

    return hf_paths[0] if G == 1 else hf_paths
