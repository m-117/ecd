import torch
from torch.utils import data


class test_data(data.Dataset):
    """Dataset class."""
    
    def __init__(self, data, opt):
        """Initialize the dataset. Data input requires shape (x, 51) with 50 expression + 1 pose parameters where x is the number of frames."""

        self.data_array = data
        self.sequences = [self.data_array[x:x+opt.seq_len] for x in range(0, self.data_array.shape[0], 1) if len(self.data_array[x:x+opt.seq_len])==opt.seq_len]

    def __getitem__(self, index):
        """Return one sequence."""

        return torch.FloatTensor(self.sequences[index])

    def __len__(self):
        """Return the number of sequences."""

        return len(self.sequences)

def get_data_loader(input_data, opt):
    """Creates and returns a new dataset object."""
    input_dataset = test_data(input_data, opt)
    return data.DataLoader(dataset=input_dataset,
                            batch_size=opt.batch_size,
                            shuffle=False,
                            num_workers=opt.nThreads,
                            pin_memory=True)
