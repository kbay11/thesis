from typing import Callable, List, Union
import tensorflow as tf
import numpy as np

class WrapperFunction:
    """
    A wrapper class designed to simplify the process of making predictions with a TensorFlow model.
    This class handles the tokenization and padding of input sequences, making it easier to work with
    raw text inputs for predictions.

    Attributes:
        model (tf.keras.Model): The TensorFlow model used for predictions.
        input_dict (dict): A dictionary mapping input tokens to their respective integer indices.
        tokenizer (callable, optional): A function for tokenizing input sequences. Defaults to a simple space-based split.
        PAD_idx (int, optional): The index used for padding shorter sequences to the model's expected input length. Defaults to 0.
        UNK_idx (int, optional): The index used for unknown tokens not found in `input_dict`. Defaults to 1.
        verbose (int, optional): Verbosity mode for TensorFlow model prediction. 0 for silent, 1 for progress bar.
    """

    def __init__(self, model: tf.keras.Model, input_dict: dict, tokenizer = lambda x: x.split(' '), PAD_idx = 0, UNK_idx = 1, verbose = 0):
        """
        Initializes the WrapperFunction2 class with the provided model, dictionaries for input and output,
        tokenizer function, padding index, unknown token index, and verbosity level.
        """
        self.model = model
        self.max_input_length = self.model.input_shape[1]  # Assumes model's input_shape is [None, length]
        self.input_dict = input_dict
        self.tokenizer = tokenizer
        self.PAD_idx = PAD_idx
        self.UNK_idx = UNK_idx
        self.verbose = verbose

    def process_sequence(self, sequence: List[str]) -> List[int]:
        """
        Maps each token in the sequence to its corresponding integer index. Unknown tokens are mapped to `UNK_idx`.

        Args:
            sequence (list of str): A tokenized sequence of words.

        Returns:
            list of int: The sequence converted to a list of integer indices.
        """
        return [self.input_dict.get(elem, self.UNK_idx) for elem in sequence]

    def pad_sequence(self, sequence: List[str]) -> List[int]:
        """
        Pads the sequence to ensure it matches the model's expected input length.

        Args:
            sequence (list of int): A list of integer indices representing a sequence.

        Returns:
            list of int: The padded sequence.
        """
        return [self.PAD_idx] * (self.max_input_length - len(sequence)) + sequence

    def __call__(self, sequences: Union[str, List[str]], verbose = 0) -> np.ndarray:
        """
        Processes and makes a prediction for the given input sequence(s) using the model. This method handles
        both single strings and lists of strings, tokenizing, converting to integer indices, padding, and predicting.

        Args:
            sequences (str or list of str): The input sequence(s) as a single string or a list of strings.
            verbose (int, optional): Verbosity mode for TensorFlow model prediction. 0 for silent, 1 for progress bar.

        Returns:
            The prediction result(s) from the model.

        Raises:
            ValueError: If the input is neither a string nor a list of strings.
        """

        if isinstance(sequences, str): # Check if string, else it should be a list of strings
            sequences = [sequences]

        sequences = [self.tokenizer(sequence) for sequence in sequences] # Tokenize string to list
        sequences = [self.process_sequence(sequence) for sequence in sequences] # Convert to integer indices
        sequences = [self.pad_sequence(sequence,) for sequence in sequences]  # Pad each sequence

        # Make and return the prediction
        return self.model.predict(sequences, verbose=verbose)