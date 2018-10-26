from google.cloud import automl_v1beta1 as automl
from enum import Enum
import gcsfs

# Set up GCP service
client = automl.AutoMlClient()


# Create Enum for classification type to prevent bad input
class ClassificationType(Enum):
    multiclass = 'MULTICLASS'
    multilabel = 'MULTILABEL'

    @classmethod
    def has_name(cls, name):
        return any(name == item.name for item in cls)


def create_dataset(
        dataset_name: str,
        project_id: str,
        region: str = 'us-central1',
        classification_type: str = 'MULTICLASS') -> object:
    """
    Create dataset for use by AutoML in GCP
    :param dataset_name: name for dataset
    :param project_id: project_id from GCP
    :param region: GCP region where dataset will be stored
    :param classification_type: MULTICLASS for one label per item,
                                MULTILABEL for 2+ labels per item
    :return:
    """
    # Test classification_type is valid
    if not ClassificationType.has_name(classification_type):
        raise ValueError(
            'classification_type must be "MULTICLASS" or "MULTILABEL"'
        )

    project_location = client.location_path(project_id, region)
    dataset_metadata = {'classification_type': classification_type}
    dataset_config = {
        'display_name': dataset_name,
        'text_classification_dataset_metadata': dataset_metadata,
    }
    new_dataset = client.create_dataset(project_location, dataset_config)

    print(f'Dataset name: {new_dataset.name}')
    print(f'Dataset id: {new_dataset.name.split("/")[-1]}')
    print(f'Dataset display name: {new_dataset.display_name}')
    print('Text classification dataset metadata:')
    print(f'\t{new_dataset.text_classification_dataset_metadata)}')
    print(f'Dataset example count: {new_dataset.example_count}')
    print('Dataset create time:')
    print(f'\tseconds: {new_dataset.create_time.seconds}')
    print(f'\tnanos: {new_dataset.create_time.nanos}')

    return new_dataset
