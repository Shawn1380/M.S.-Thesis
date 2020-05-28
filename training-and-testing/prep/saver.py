""" This module includes functions to save well-trained keras model. """

from prep import constants
import pathlib
import keras

def save_model(model, NN_type, num_of_cells, num_of_CUEs, num_of_D2Ds = None):
    """ Whole-model saving (configuration and weights).

    Whole-model saving means creating a file that will contain:
        1. The architecture of the model, allowing to re-create the model.
        2. The weights of the model.
        3. The training configuration (loss, optimizer).
        4. The state of the optimizer, allowing to resume training exactly where you left off.

    # Aruguments:

        NN_type: string
            Type of neural network.
        num_of_cells: int
            Number of the cells in the cellular system.
        num_of_CUEs: int
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
    assert num_of_CUEs in constants.CUE_range, f"The 'num_of_CUEs' must be element in {constants.CUE_range}."
    assert num_of_D2Ds in constants.D2D_range or num_of_D2Ds is None, f"The 'num_of_D2Ds' must be element in {constants.D2D_range}."
    
    # Get the path to the file to save the model to
    model_dir = pathlib.Path.cwd().joinpath('model')
    cell_dir = '{} cell'.format(num_of_cells)
    model_dir = model_dir.joinpath(cell_dir)

    if num_of_D2Ds:
        file_name = 'model_Cell_{}_CUE_{}_D2D_{}_{}'.format(num_of_cells, num_of_CUEs, num_of_D2Ds, NN_type)
    else:
        file_name = 'model_Cell_{}_CUE_{}_{}'.format(num_of_cells, num_of_CUEs, NN_type)

    file_path = model_dir.joinpath(file_name)

    # Save the model in HDF5 format
    model.save(file_path, overwrite = False, include_optimizer = True)

def save_weights(model, NN_type, num_of_cells, num_of_CUEs, num_of_D2Ds = None):
    """ Weights-only saving.

    Weights-only saving means creating a file that will contain the weights of the model.

    # Aruguments:

        NN_type: string
            Type of neural network.
        num_of_cells: int
            Number of the cells in the cellular system.
        num_of_CUEs: int
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
    assert num_of_CUEs in constants.CUE_range, f"The 'num_of_CUEs' must be element in {constants.CUE_range}."
    assert num_of_D2Ds in constants.D2D_range or num_of_D2Ds is None, f"The 'num_of_D2Ds' must be element in {constants.D2D_range}."

def save_configuration(model, NN_type, num_of_cells, num_of_CUEs, num_of_D2Ds = None):
    """ Configuration-only saving.

    Weights-only saving means creating a file that will contain the architecture of the model,
    and not its weights or its training configuration.

    # Aruguments:

        NN_type: string
            Type of neural network.
        num_of_cells: int
            Number of the cells in the cellular system.
        num_of_CUEs: int
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
    assert num_of_CUEs in constants.CUE_range, f"The 'num_of_CUEs' must be element in {constants.CUE_range}."
    assert num_of_D2Ds in constants.D2D_range or num_of_D2Ds is None, f"The 'num_of_D2Ds' must be element in {constants.D2D_range}."