import os

import torch

from torch_model.runner import train


def _check_cuda_memory(device_id):
    """
    Given a cuda device number, return free memory
    :param device_id: an integer that maps to which cuda gpu
    :return: memory available in bytes
    """
    memory_total = torch.cuda.get_device_properties(device_id).total_memory
    memory_allocated = torch.cuda.memory_allocated(device_id)
    return memory_total - memory_allocated


def select_best_device(device=None):
    """
    Pick the best device in this order: cuda, mps (apple), cpu
    :return: torch device to use
    """

    # override the logic if we pass in a device
    if device:
        return torch.device(device)

    # pick the best gpu, if it's available
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        best_gpu = max(range(num_gpus), key=lambda i: _check_cuda_memory(i))
        return torch.device(f"cuda:{best_gpu}")

    # pick apple mps next, if it's available
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")

    # return cpu as a last resort
    return torch.device("cpu")


def _create_folders_if_not_exist(folder_path):
    """
    private function to create a folder if it doesn't already exist
    :param folder_path: the folder and path to it
    """
    try:
        os.makedirs(folder_path, exist_ok=True)
        os.makedirs(os.path.join(folder_path, "debug_images"), exist_ok=True)
        print(f"Directory '{folder_path}' created successfully or already exists.")
    except Exception as e:
        print(f"Error creating directory '{folder_path}': {e}")


def create_directories(results_directory):
    """
    code currently will throw a fit if debug_images folder isn't already populated
    """
    _create_folders_if_not_exist(results_directory)


if __name__ == "__main__":
    device = select_best_device()
    print(f"Using {device} device...")

    # file_path = "datasets/sprite_gpt4aug.npy"
    file_path = "datasets/data_train.npy"

    hyperparameters = {"epochs": 100, "batch_size": 128, "learning_rate": 0.0001}
    results_directory = f"./results/model_epochs_{hyperparameters['epochs']}_batch_{hyperparameters['batch_size']}_lr_{hyperparameters['learning_rate']}"
    create_directories(results_directory)

    train(file_path, hyperparameters, device, results_directory)
