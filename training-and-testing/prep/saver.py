""" This module includes functions to save well-trained keras model. """

from prep import constants
import pathlib
import keras

def save_model(model, NN_type, num_of_cells, num_of_CUEs = None, num_of_D2Ds = None):
    """ Whole-model saving (configuration and weights).

    Whole-model saving means creating a file (HDF5) that will contain:
        1. The architecture of the model, allowing to re-create the model.
        2. The weights of the model.
        3. The training configuration (loss, optimizer).
        4. The state of the optimizer, allowing to resume training exactly where you left off.

    # Aruguments:

        model: keras.engine.sequential.Sequential
            A keras model instance (compiled).
        NN_type: string
            Type of neural network.
        num_of_cells: int
            Number of the cells in the cellular system.
        num_of_CUEs: int, optional
            Number of the CUEs in each cell.
        num_of_D2Ds: int, optional
            Number of the D2D pairs in each cell.

    # Return:

        None
    """

    # Insert debugging assertions
    assert type(model) is keras.engine.sequential.Sequential, "The 'model' must be sequential model."
    assert type(NN_type) is str, "The 'NN_type' must be string."
    assert num_of_cells in constants.cell_range, f"The 'num_of_cells' must be element in {constants.cell_range}."
    assert num_of_CUEs in constants.CUE_range or num_of_CUEs is None, f"The 'num_of_CUEs' must be element in {constants.CUE_range}."
    assert num_of_D2Ds in constants.D2D_range or num_of_D2Ds is None, f"The 'num_of_D2Ds' must be element in {constants.D2D_range}."
    
    # Get the path to the file to save the model to
    model_dir = pathlib.Path.cwd().joinpath('model')
    cell_dir = f'{num_of_cells}-cell'
    model_dir = model_dir.joinpath(cell_dir)

    if num_of_CUEs and num_of_D2Ds:
        file_name = f'model_Cell_{num_of_cells}_CUE_{num_of_CUEs}_D2D_{num_of_D2Ds}_{NN_type}.h5'
    else:
        file_name = f'model_Cell_{num_of_cells}_{NN_type}.h5'

    file_path = str(model_dir.joinpath(file_name))

    # Save the model in HDF5 format
    model.save(file_path, overwrite = False, include_optimizer = True)

def save_weights(model, NN_type, num_of_cells, num_of_CUEs = None, num_of_D2Ds = None):
    """ Weights-only saving.

    Weights-only saving means creating a file (HDF5) that will contain the weights of the model.

    # Aruguments:

        model: keras.engine.sequential.Sequential
            A keras model instance (compiled).
        NN_type: string
            Type of neural network.
        num_of_cells: int
            Number of the cells in the cellular system.
        num_of_CUEs: int, optional
            Number of the CUEs in each cell.
        num_of_D2Ds: int, optional
            Number of the D2D pairs in each cell.

    # Return:

        None
    """

    # Insert debugging assertions
    assert type(model) is keras.engine.sequential.Sequential, "The 'model' must be sequential model."
    assert type(NN_type) is str, "The 'NN_type' must be string."
    assert num_of_cells in constants.cell_range, f"The 'num_of_cells' must be element in {constants.cell_range}."
    assert num_of_CUEs in constants.CUE_range or num_of_CUEs is None, f"The 'num_of_CUEs' must be element in {constants.CUE_range}."
    assert num_of_D2Ds in constants.D2D_range or num_of_D2Ds is None, f"The 'num_of_D2Ds' must be element in {constants.D2D_range}."

    # Get the path to the file to save the weights to
    model_dir = pathlib.Path.cwd().joinpath('model')
    cell_dir = f'{num_of_cells}-cell'
    model_dir = model_dir.joinpath(cell_dir)

    if num_of_CUEs and num_of_D2Ds:
        file_name = f'weights_Cell_{num_of_cells}_CUE_{num_of_CUEs}_D2D_{num_of_D2Ds}_{NN_type}.h5'
    else:
        file_name = f'weights_Cell_{num_of_cells}_{NN_type}.h5'

    file_path = str(model_dir.joinpath(file_name))

    # Save the weights in HDF5 format
    model.save_weights(file_path, overwrite = False)


def save_configuration(model, NN_type, num_of_cells, num_of_CUEs = None, num_of_D2Ds = None):
    """ Configuration-only saving.

    Weights-only saving means creating a file (JSON) that will contain the architecture of the model,
    and not its weights or its training configuration.

    # Aruguments:

        model: keras.engine.sequential.Sequential
            A keras model instance (compiled).
        NN_type: string
            Type of neural network.
        num_of_cells: int
            Number of the cells in the cellular system.
        num_of_CUEs: int, optional
            Number of the CUEs in each cell.
        num_of_D2Ds: int, optional
            Number of the D2D pairs in each cell.

    # Return:

        None
    """

    # Insert debugging assertions
    assert type(model) is keras.engine.sequential.Sequential, "The 'model' must be sequential model."
    assert type(NN_type) is str, "The 'NN_type' must be string."
    assert num_of_cells in constants.cell_range, f"The 'num_of_cells' must be element in {constants.cell_range}."
    assert num_of_CUEs in constants.CUE_range or num_of_CUEs is None, f"The 'num_of_CUEs' must be element in {constants.CUE_range}."
    assert num_of_D2Ds in constants.D2D_range or num_of_D2Ds is None, f"The 'num_of_D2Ds' must be element in {constants.D2D_range}."

    # Get the path to the file to save the configuration to
    model_dir = pathlib.Path.cwd().joinpath('model')
    cell_dir = f'{num_of_cells}-cell'
    model_dir = model_dir.joinpath(cell_dir)

    if num_of_CUEs and num_of_D2Ds:
        file_name = f'configuration_Cell_{num_of_cells}_CUE_{num_of_CUEs}_D2D_{num_of_D2Ds}_{NN_type}.json'
    else:
        file_name = f'configuration_Cell_{num_of_cells}_{NN_type}.json'

    file_path = str(model_dir.joinpath(file_name))

    # Save the configuration in JSON format
    configuration = model.to_json()

    with open(file_path, 'w') as json_file:
        json_file.write(configuration)