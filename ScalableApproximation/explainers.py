from abc import ABC, abstractmethod
from typing import List, Tuple, Union, Any
import numpy as np
from shap._explanation import Explanation
from WrapperFunction import WrapperFunction
from utils2 import _mask_input, _generate_bitmasks, _insert, _insert_np, _factorial, _combinations, _n_k_bitmasks, _n_k_coefficient

class Explainer(ABC):
    """
    Serves as an abstract base class for different explanation methods applied to TensorFlow models.
    This class defines a common interface for generating explanations of model predictions, 
    emphasizing the contribution of individual input features. It relies on an instance of 
    WrapperFunction to preprocess textual data for the model.

    Attributes:
        function (WrapperFunction): An instance of WrapperFunction that encapsulates the model and its preprocessing.
        tokenizer (Callable): A tokenizer function derived from the WrapperFunction instance used for text processing.

    Methods:
        explain: Must be implemented by subclasses to provide specific explanation mechanisms based on the model's output.
    """

    def __init__(self, wrapper_function: WrapperFunction) -> None:
        """
        Initializes the Explainer with a specific WrapperFunction instance.

        Parameters:
            wrapper_function (WrapperFunction): An instance of WrapperFunction that provides model interaction and preprocessing.

        Raises:
            TypeError: If `wrapper_function` is not an instance of `WrapperFunction`.
        """
        if not isinstance(wrapper_function, WrapperFunction):
            raise TypeError("wrapper_function must be an instance of WrapperFunction")
        self.function = wrapper_function
        self.tokenizer = wrapper_function.tokenizer

    @abstractmethod
    def explain(self, sequence: str, output_names: List[str], mask: str, exp_object: bool = True) -> Any:
        """
        Abstract method that subclasses must implement to provide an explanation based on the model's prediction.
        
        This method processes the given sequence using the specified explanation method and returns the results 
        in a format determined by the `exp_object` flag.

        Parameters:
            sequence (str): The input text sequence to explain.
            output_names (List[str]): Names of the model outputs for which explanations are sought.
            mask (str): The token used for masking parts of the input sequence.
            exp_object (bool, optional): Determines the format of the explanation output. Defaults to True.

        Returns:
            Any: The explanation result, which may vary in format depending on the subclass implementation and `exp_object` parameter.
        """
        pass

class BaselineExplainer(Explainer):
    """
    Implements a baseline explanation method for a TensorFlow model wrapped by a WrapperFunction instance.
    This explainer evaluates the impact of sequentially adding each feature in the input sequence on the model's output,
    providing a simple yet intuitive understanding of each feature's contribution.

    Inherits from:
        Explainer (ABC): An abstract base class defining the interface for model explanation methods.
    """

    def __init__(self, wrapper_function: WrapperFunction) -> None:
        """
        Initializes the BaselineExplainer with a specific WrapperFunction instance.

        Parameters:
            wrapper_function (WrapperFunction): An instance of WrapperFunction that provides model interaction and preprocessing.
        """
        super().__init__(wrapper_function)

    def __call__(self, sequence: str, output_names: List[str], mask: str = '[PAD]', exp_object: bool = True) -> Union[Explanation, Tuple[np.ndarray, np.ndarray]]:
        """
        A convenience method that directly invokes the explain method.

        Parameters:
            sequence (str): The input text sequence to explain.
            output_names (List[str]): Names of the model outputs for which explanations are sought.
            mask (str, optional): The token used for masking parts of the input sequence. Defaults to '[PAD]'.
            exp_object (bool, optional): Determines the format of the explanation output. Defaults to True.

        Returns:
            Union[Explanation, Tuple[np.ndarray, np.ndarray]]: The explanation result, either as an Explanation object or a tuple of numpy arrays, based on the exp_object flag.
        """
        return self.explain(sequence, output_names, mask=mask, exp_object=exp_object)

    def explain(self, sequence: str, output_names: List[str], mask: str = '[PAD]', exp_object: bool = True) -> Union[Explanation, Tuple[np.ndarray, np.ndarray]]:
        """
        Implements the explanation logic by evaluating the model's response to the sequential inclusion of each feature in the input sequence.

        Parameters:
            sequence (str): The input text sequence to explain.
            output_names (List[str]): Names of the model outputs for which explanations are sought.
            mask (str, optional): The token used for masking parts of the input sequence. Defaults to '[PAD]'.
            exp_object (bool, optional): Determines the format of the explanation output. Defaults to True.

        Returns:
            Union[Explanation, Tuple[np.ndarray, np.ndarray]]: The calculated differences in model output as features are sequentially added, either wrapped in an Explanation object or as raw values.
        """
        sequence = sequence.split(' ')
        base_values = self.function([mask])

        # Generates inputs for the model by progressively adding features to the masked sequence
        inputs = [mask] + [" ".join(sequence[:i+1]) for i in range(len(sequence))]
        output = self.function(inputs)

        # Computes the difference in model output as each feature is added
        values = np.diff(output, axis=0)
        values = np.expand_dims(values, axis=0)

        if exp_object:
            return Explanation(values=values.astype(np.float64),
                               base_values=base_values.astype(np.float64),
                               data=(np.array(sequence, dtype=object),),
                               output_names=output_names,
                               feature_names=[sequence])
        else:
            return values, base_values

class ShapleyValueExplainer(Explainer):
    """
    Implements Shapley values to explain predictions of a TensorFlow model wrapped by a WrapperFunction.
    Shapley values offer a powerful and fair method to attribute the prediction output to each input feature,
    based on the concept from cooperative game theory.
    
    Inherits from:
        Explainer: An abstract base class for implementing different explanation methodologies.
    """

    def __init__(self, wrapper_function: WrapperFunction) -> None:
        """
        Initializes the ShapleyValueExplainer with a specific WrapperFunction instance.

        Parameters:
            wrapper_function (WrapperFunction): The instance of WrapperFunction for interacting with the model.
        """
        super().__init__(wrapper_function)

    def __call__(self, sequence: str, output_names: List[str], mask: str = '[PAD]', collapse: bool = True, exp_object: bool = True) -> Union[Explanation, tuple]:
        """
        Convenience method to directly invoke the explain method.

        Parameters:
            sequence (str): The input text sequence to explain.
            output_names (List[str]): The names of the model outputs for which explanations are sought.
            mask (str, optional): The token used for masking parts of the input sequence. Defaults to '[PAD]'.
            collapse (bool, optional): Specifies if the masked inputs should be collapsed into a single string. Defaults to True.
            exp_object (bool, optional): Determines if the explanation should be returned as an object. Defaults to True.

        Returns:
            Union[Explanation, tuple]: The explanation result, formatted based on the exp_object flag.
        """
        return self.explain(sequence, output_names, mask=mask, collapse=collapse, exp_object=exp_object)

    def explain(self, sequence: str, output_names: List[str], mask: str = '[PAD]', collapse: bool = True, exp_object: bool = True) -> Union[Explanation, tuple]:
        """
        Computes Shapley values for each feature in the input sequence.

        Parameters:
            sequence (str): The input text sequence to explain.
            output_names (List[str]): The names of the model outputs for which explanations are sought.
            mask (str, optional): The token used for masking parts of the input sequence. Defaults to '[PAD]'.
            collapse (bool, optional): Specifies if the masked inputs should be collapsed into a single string. Defaults to True.
            exp_object (bool, optional): Determines if the explanation should be returned as an object. Defaults to True.

        Returns:
            Union[Explanation, tuple]: The calculated Shapley values, either wrapped in an Explanation object or as raw values.
        """
        sequence = sequence.split(' ')
        base_values = self.function([' '.join([mask] * len(sequence))])

        # Initialize lists to hold padded sequences and their associated factors
        s_pluses, s_minuses, factors = [], [], []

        # Generate padded sequences and save them
        for idx, _ in enumerate(sequence):
            bitmasks = _generate_bitmasks(len(sequence) - 1)
            for bitmask in bitmasks:
                s_plus = _insert(bitmask, idx, 1)
                s_minus = _insert(bitmask, idx, 0)

                s_plus = " ".join(_mask_input(sequence, s_plus, mask_token=mask, collapse=collapse))
                s_minus = " ".join(_mask_input(sequence, s_minus, mask_token=mask, collapse=collapse))

                S = sum(bitmask)
                n_S_1 = len(sequence) - S - 1

                factor = _factorial(S) * _factorial(n_S_1) / _factorial(len(sequence))
                s_pluses.append(s_plus)
                s_minuses.append(s_minus)
                factors.append(factor)

        # Feed all the generated padded sequences into the model at once for efficiency 
        res = self.function(s_pluses + s_minuses)

        # Calculate the coeffiecients of each padded subsequence
        factors = np.array(factors)[:, np.newaxis]

        # Calculate the marginal contribution of each element
        res = (res[:len(s_pluses)] - res[len(s_pluses):]) * factors

        # Iterate thorugh res in chunks to extract the Shapley value for each element of the sequence
        chunk_size = res.shape[0] // len(sequence)
        num_chunks = res.shape[0] // chunk_size
        values = np.zeros((num_chunks, res.shape[1]))

        for i in range(num_chunks):
            chunk = res[i * chunk_size: (i + 1) * chunk_size]
            values[i] = np.sum(chunk, axis=0)
        
        # Add new axis for compatibility with Explanation object
        values = values[np.newaxis,::]

        if exp_object:
            return Explanation(values=values.astype(np.float64),
                            base_values=base_values.astype(np.float64),
                            data=(np.array(sequence, dtype=object),),
                            output_names=output_names,
                            feature_names=[sequence])
        else:
            return values, base_values