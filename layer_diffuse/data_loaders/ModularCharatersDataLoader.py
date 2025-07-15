import torch
from datasets import load_dataset
from datasets.iterable_dataset import IterableDataset
from typing import Iterator, Any
from torchvision import transforms
import os
from PIL import Image
from numpy import random


class ShuffleDataset(torch.utils.data.IterableDataset):
    def __init__(self, dataset, buffer_size):
        super().__init__()
        self.dataset = dataset
        self.buffer_size = buffer_size

    def __iter__(self) -> Iterator[Any]:
        shuffle_buffer = []
        dataset_iter = None
        try:
            dataset_iter = iter(self.dataset)
            for i in range(self.buffer_size):
                shuffle_buffer.append(next(dataset_iter))
        except:
            self.buffer_size = len(shuffle_buffer)

        try:
            while True:
                try:
                    if dataset_iter is not None:
                        item = next(dataset_iter)
                        random_index = random.randint(0, self.buffer_size - 1)
                        yield shuffle_buffer[random_index]
                        shuffle_buffer[random_index] = item
                except StopIteration:
                    break
            while len(shuffle_buffer) > 0:
                yield shuffle_buffer.pop()
        except GeneratorExit:
            pass

    def __getitem__(self, index) -> Any:
        return super().__getitem__(index)


class ModularCharactersDataLoader(torch.utils.data.DataLoader):
    def __init__(
        self,
        dataset_name: str,
        split: str,
        image_size: int,
        batch_size: int = 16,
        shuffle: bool = True,
        num_workers: int = 0,
        pin_memory: bool = False,
        persistent_workers: bool = False,
        conversionRGBA: bool = False,
        streaming: bool = False,
        vocab: dict = {},
    ):
        if not os.path.exists("cache/datasets"):
            os.makedirs("cache/datasets", exist_ok=True)
        dataset = load_dataset(
            dataset_name, split=split, cache_dir="cache/datasets", streaming=streaming
        )
        self.dataset_name = dataset_name
        self.split = split
        if not vocab:
            vocab = dataset["prompt"]  # type: ignore
            vocab = list(dict.fromkeys(dataset["prompt"]))  # type: ignore
            vocab = sorted(vocab)  # type: ignore
            self.vocab = dict(zip(vocab, range(len(vocab))))
        else:
            self.vocab = vocab

        preprocess = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                # transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

        def transform(rows: dict) -> dict:
            # Convert RGBA images to RGB with white background
            def rgba_to_rgb_white(img):
                if img.mode == "RGBA" and conversionRGBA:
                    background = Image.new("RGB", img.size, (255, 255, 255))
                    background.paste(img, mask=img.split()[3])  # 3 is the alpha channel
                    return background
                return img

            rows["input"] = [rgba_to_rgb_white(image) for image in rows["input"]]
            rows["target"] = [rgba_to_rgb_white(image) for image in rows["target"]]
            images_input = [preprocess(image) for image in rows["input"]]
            images_target = [preprocess(image) for image in rows["target"]]
            class_labels = [
                torch.tensor(self.vocab.get(prompt, -1), dtype=torch.long).unsqueeze(0)
                for prompt in rows["prompt"]
            ]
            return {
                "input": images_input,
                "target": images_target,
                "label": class_labels,
            }

        if isinstance(dataset, IterableDataset):
            dataset = dataset.map(transform, batched=True, remove_columns=["prompt"])  # type: ignore
            dataset = ShuffleDataset(dataset, buffer_size=1024) if shuffle else dataset
            return super().__init__(
                dataset,  # type: ignore
                batch_size=batch_size,
                num_workers=num_workers,
                pin_memory=pin_memory,
                persistent_workers=persistent_workers,
            )
        else:
            dataset.set_transform(transform)  # type: ignore

            return super().__init__(
                dataset,  # type: ignore
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                pin_memory=pin_memory,
                persistent_workers=persistent_workers,
            )


def get_modular_char_dataloader(
    dataset_name: str,
    split: str,
    image_size: int,
    batch_size: int = 16,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False,
    persistent_workers: bool = False,
    streaming: bool = False,
    conversionRGBA: bool = False,
    vocab: dict = {},
):
    return ModularCharactersDataLoader(
        dataset_name=dataset_name,
        split=split,
        image_size=image_size,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        streaming=streaming,
        conversionRGBA=conversionRGBA,
        vocab=vocab,
    )
