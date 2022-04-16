import os
import yaml
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient


def load_conf():
    dir_root = os.path.dirname(os.path.abspath(__file__))
    with open(dir_root + '/config.yaml', 'r')as yamlfile:
        return yaml.safe_load(yamlfile)


def get_files(dir):
    with os.scandir(dir) as entries:
        for entry in entries:
            yield entry


config = load_conf()
#files = get_files(config['source_folder'] + '/input_data')
#print(*files)

blob_service_client = BlobServiceClient.from_connection_string(config['azure_storage_connectionstring'])