import torch
import torch.utils.data as data_utils
from torchvision import transforms
from pathlib import Path
from PIL import Image

class PathFinderDataset(torch.utils.data.Dataset):
    """Path Finder dataset."""

    # There's an empty file in the dataset
    blacklist = {"pathfinder32/curv_baseline/imgs/0/sample_172.png"}

    def __init__(self, data_dir="/Data/pgi-15/datasets/lra_release/pathfinder32/",
                transform=None):
        """
        Args:
            data_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data_dir = Path(data_dir).expanduser()
        assert self.data_dir.is_dir(), f"data_dir {str(self.data_dir)} does not exist"
        self.transform = transform
        samples = []
        # for diff_level in ['curv_baseline', 'curv_contour_length_9', 'curv_contour_length_14']:
        for diff_level in ["curv_baseline"]:
            path_list = sorted(
                list((self.data_dir / diff_level / "metadata").glob("*.npy")),
                key=lambda path: int(path.stem),
            )
            assert path_list, "No metadata found"
            for metadata_file in path_list:
                with open(metadata_file, "r") as f:
                    for metadata in f.read().splitlines():
                        metadata = metadata.split()
                        image_path = Path(diff_level) / metadata[0] / metadata[1]
                        if (
                            str(Path(self.data_dir.stem) / image_path)
                            not in self.blacklist
                        ):
                            label = int(metadata[3])
                            samples.append((image_path, label))
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, target = self.samples[idx]
        # https://github.com/pytorch/vision/blob/9b29f3f22783112406d9c1a6db47165a297c3942/torchvision/datasets/folder.py#L247
        with open(self.data_dir / path, "rb") as f:
            sample = Image.open(f).convert("L")  # Open in grayscale
        if self.transform is not None:
            sample = self.transform(sample)
        #import pdb
        #pdb.set_trace()
        return sample.flatten(start_dim=1).permute(1,0), target #.permute(1,0)
    
def make_loader(args, Dataset, train=True):
    # Training batch size is definined by argument. 
    # Testing batch size is defined by argument if the argument is not greater than the size of the test dataset. Otherwise, the test batch size is equal to the length of the test dataset.
    batch_size = args.batch_size if train else args.test_batch_size
    # Create data-loader generator object
    shuffle = args.shuffle_dataset if train else False
    return data_utils.DataLoader(Dataset,
                                 batch_size=batch_size,
                                 shuffle=shuffle,
                                 drop_last=True)

if __name__ == '__main__':

    class ARGS():
        def __init__(self) -> None:
            self.batch_size=32
            self.shuffle_dataset=True
            
    args = ARGS()
    Dataset = PathFinderDataset(transform=transforms.ToTensor())

    dataset = make_loader(args, Dataset)
    for i, (x, y) in enumerate(dataset):
        print("input shape", x.size())
        print("out shape", y.size())
        print("out shape", y)
        exit()
