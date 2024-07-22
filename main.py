from models.general_dollarmodel import DollarModel, train

if __name__ == "__main__":
    input_shape = (10, 10, 16)
    epochs = 100
    batch_size = 256
    encpic = DollarModel(model_name="test_modelname", img_shape=input_shape, lr=0.0005, embedding_dim=384, z_dim=5, filter_count=128, kern_size=5, num_res_blocks=3, dataset_type='map', data_path='datasets/maps_noaug.npy')
    train(encpic, epochs, batch_size, sample_interval=10)
