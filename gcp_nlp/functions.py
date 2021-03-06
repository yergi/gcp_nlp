from google.cloud import automl_v1beta1 as automl
from google.cloud.automl_v1beta1 import enums
import gcsfs

# Set up GCP service
client = automl.AutoMlClient()


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
    if any(
            classification_type.upper() == valid_type.name
            for valid_type in enums.ClassificationType
    ):
         classification_type = classification_type.upper()
    else:
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


def import_data(
        project_id: str,
        dataset_id: str,
        path: str,
        region: str = 'us-central1'
) -> str:
    """
    Import labelled items.
    :param project_id: project_id from GCP
    :param dataset_id: Dataset id from GCP
    :param path: path to CSV file containing labelled items
    :param region: GCP region where function will be performed
    :return: String of result status
    """

    dataset_loc = client.dataset_path(
        project_id, region, dataset_id
    )

    # TODO: Update to handle path as str or list
    input_uris = path.split(',')
    input_config = {'gcs_source': {
        'input_uris': input_uris
    }}

    response = client.import_data(dataset_loc, input_config)

    print('Processing...')
    print(f'Process Complete. {response.result()}')

    return response.result()


def upload_data(
        project_name: str,
        lpath: str,
        rpath: str
) -> None:
    """
    Wrapper around GCSFS to upload CSV file into Google Cloud Storage
    :param project_name: Name of project in GCP
    :param lpath: local filepath
    :param rpath: remote filepath
    :return: None
    """
    fs = gcsfs.GCSFileSystem(project=project_name)
    fs.put(lpath, rpath)


def create_model(
        project_id: str,
        dataset_id: str,
        model_name: str,
        region: str = 'us-central1'
) -> str:
    """
    Create NLP model and begin training
    :param project_id: Project ID in GCP
    :param dataset_id: Dataset ID in GCP
    :param model_name: Name for model
    :param region: Region where GCP will run model
    :return: string of operation name
    """

    project_loc = client.location_path(project_id, region)

    model_config = {
        'display_name': model_name,
        'dataset_id': dataset_id,
        'text_classification_model_metadata': {},
    }

    response = client.create_model(project_loc, model_config)
    print(f'Training operation name: {response.operation.name}')
    print('Training started...')

    return response.operation.name


def get_operation_status(
        operation_id: str
) -> object:
    """
    Return status of given model operation
    :param operation_id: ID of operation, which is the output of the *_model functions
    :return: Operation object with status details
    """

    status = client.transport._operations_client.get_operation(
        operation_id
    )

    print(f'Current status: {status}')

    return status


def get_model(
        project_id: str,
        model_id,
        region: str = 'us-central1'
) -> object:
    """
    Retrieve model from GCP
    :param project_id: GCP project ID
    :param model_id: GCP model ID
    :param region: GCP region where operations will be run
    :return: Model object from GCP
    """

    # Get model details
    model_path = client.model_path(project_id, region, model_id)
    model = client.get_model(model_path)

    # Retrieve deployment state
    deployment_state = ''
    if model.deployment_state == enums.Model.DeploymentState.DEPLOYED:
        deployment_state = 'deployed'
    else:
        deployment_state = 'undeployed'

    # Print model details
    print(f'Model name: {model.name}')
    print(f'Model ID: {model.name.split("/")[-1]}')
    print(f'Model display name: {model.display_name}')
    print('Model create time:')
    print(f'\tseconds: {model.create_time.seconds}')
    print(f'\tnanos: {model.create_time.nanos}')
    print(f'Deployment state: {deployment_state}')

    return model


def list_model_evaluations(
        project_id: str,
        model_id: str,
        filter_: str,
        region: str = 'us-central1'
) -> object:
    """
    List model evaluations to determine quality
    :param project_id: GCP project ID
    :param model_id: GCP model ID
    :param filter_: Filter expression to limit evals returned
    :param region: GCP compute region
    :return: object of evaluations
    """

    model_path = client.model_path(project_id, region, model_id)

    # List all of the model evals, apply filter if it exists to limit results
    evaluations = client.list_model_evaluations(model_path, filter_)

    print('List of model evaluations:')
    for evaluation in evaluations:
        print(evaluation)

    return evaluations


def get_model_evaluation(
        project_id: str,
        model_id: str,
        model_evaluation_id: str,
        region: str = 'us-central1'
) -> object:
    """
    Get specific model evaluation
    :param project_id: GCP project ID
    :param model_id: GCP model ID
    :param model_evaluation_id: GCP model evaluation ID
    :param region: GCP compute region
    :return: object with evaluation results
    """

    model_evaluation_path = client.model_evaluation_path(
        project_id, region, model_id, model_evaluation_id
    )

    evaluation = client.get_model_evaluation(model_evaluation_path)

    print(evaluation)

    return evaluation


def predict(
        project_id: str,
        model_id: str,
        target_data: str,
        mime_type: str = 'text/plain',
        region: str = 'us-central1',
        params: dict = {}
) -> object:
    """
    Return prediction output of model on single text value
    :param project_id: GCP project ID
    :param model_id: GCP model ID
    :param target_data: string to input into model for prediction
    :param mime_type: MIME type of string, if not plain text
    :param region: GCP compute region
    :param params: Empty dict
    :return: Object of prediction result
    """

    prediction_client = automl.PredictionServiceClient()

    model_path = client.model_path(
        project_id, region, model_id
    )

    payload = {
        'text_snippet': {
            'content': target_data,
            'mime_type': mime_type
        }
    }

    prediction_result = prediction_client.predict(
        model_path, payload, params
    )

    return prediction_result
