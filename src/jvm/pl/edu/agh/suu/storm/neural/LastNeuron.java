package pl.edu.agh.suu.storm.neural;

import java.io.Serializable;
import java.util.HashMap;
import java.util.Map;

/**
 * This class represents a single neuron in the output layer of the neural network.
 */
public class LastNeuron implements Serializable {

	private static final long serialVersionUID = 7931056296740593474L;
	
	private int inputSize;
	private double[] weights; // length of weights is inputSize + 1 (bias)
	private double[] forwardInput; // length of forwardInput is inputSize (no bias)
	private double alpha; // convergence coefficient
	private double delta = 0.0d;
	private double[] derivatives;
	private Map<Integer, Double> expectedValues;
	private double result;
	
	/**
	 * Creates a new neuron.
	 * @param inputSize size of the previous layer
	 * @param weights initial neuron weights;
	 *        size of the array should be equal to inputSize + 1; first value represent bias unit weight
	 * @param alpha neural network learning rate
	 */
	public LastNeuron(int inputSize, double[] weights, double alpha) {
		this.inputSize = inputSize;
		this.weights = weights;
		this.alpha = alpha;
		this.expectedValues = new HashMap<Integer, Double>();
	}
	
	/**
	 * Performs the forward propagation (output value computation).
	 * @param input input from the previous layer; size of the array should be equal to inputSize specified in the constructor
	 * 	      ({@link #LastNeuron(int, double[], double)})
	 * @return
	 */
	public double propagateForward(double[] input) { // input - neurons from previous layer
		this.forwardInput = input; // saved for computing derivative in backward propagation later
		double sum = weights[0]; // bias
		for (int i = 0; i < inputSize; ++i) {
			sum += input[i] * weights[i + 1];
		}
		result = Functions.sigmoid(sum);
		return result;
	}
	
	/**
	 * Performs the backward propagation. This method does not perform the neuron weights update itself.
	 * After passing all training data through forward and backward propagation (when iteration is over)
	 * one should call {@link #updateWeights(int)} to recalculate neuron weights. 
	 * @param numberOfElement number of the element from the training set; expected value for the specified element should be
	 *        passed to this object through {@link #setExpectedValue(double, int)}, {@link #setExpectedValues(double[], int)} or
	 *        {@link #setExpectedValues(int[], double[], int)} before invoking this method
	 * @return backward propagation result; size of the array is equal to inputSize specified in the constructor
	 * 	       ({@link #LastNeuron(int, double[], double)})
	 */
	public double[] propagateBackward(int numberOfElement) {
		double delta = result - expectedValues.get(numberOfElement);
		double[] returnValues = new double[inputSize];
		for (int i = 0; i < inputSize; ++i) {
			returnValues[i] = weights[i + 1] * delta * forwardInput[i] * (1 - forwardInput[i]);
			derivatives[i] += forwardInput[i] * delta;
		}
		this.delta += delta;
		return returnValues;
	}
	
	/**
	 * Updates the neuron weights after finishing the iteration.
	 * @param trainingSetSize training set element count
	 * @see #propagateBackward(int)
	 */
	public void updateWeights(int trainingSetSize) {
		weights[0] = weights[0] - alpha * (delta / trainingSetSize); // bias
		for (int i = 0; i < inputSize; ++i) {
			weights[i + 1] = weights[i + 1] - alpha * (derivatives[i] / trainingSetSize);
		}
		resetDelta();
	}
	
	/**
	 * Alters neural network learning rate.
	 * @param newAlpha new neural network learning rate
	 */
	public void alterAlpha(double newAlpha) {
		this.alpha = newAlpha;
	}
	
	/**
	 * Should be called at the beginning of an iteration to perform necessary cleanup.
	 */
	public void resetDelta() {
		this.delta = 0.0d;
		this.derivatives = new double[inputSize];
		for (int i = 0; i < inputSize; ++i) {
			this.derivatives[i] = 0.0d;
		}
	}
	
	/**
	 * Sets the expected output value of the neural network for the specified element from the training set.
	 * @param value expected value
	 * @param position position of the element in the training set
	 */
	public void setExpectedValue(double value, int position) { // allows streaming
		this.expectedValues.put(position, value);
	}
	
	/**
	 * Sets the expected output values of the neural network for the initial values from the training set.
	 * @param values expected values; size of the array should be equal to {@link count}
	 * @param count number of the initial values from the training set
	 */
	public void setExpectedValues(double[] values, int count) {
		for (int i = 0; i < count; ++i) {
			this.expectedValues.put(i, values[i]);
		}
	}
	
	/**
	 * Sets the expected output values of the neural network for the specified values from the training set.
	 * @param keys 
	 * @param values expected values; size of the array should be equal to {@link count}
	 * @param count positions of the values in the training set; size of the array should be equal to {@link count}
	 */
	public void setExpectedValues(int[] keys, double[] values, int count) {
		for (int i = 0; i < count; ++i) {
			this.expectedValues.put(keys[i], values[i]);
		}
	}
	
	/**
	 * Cleans the expected output value of the neural network for the specified element from the training set.
	 * @param position position of the element in the training set
	 */
	public void cleanExpectedValue(int position) {
		this.expectedValues.remove(position);
	}
	
	public double[] getWeights() {
		return weights;
	}
	
	/**
	 * Returns neuron weights.
	 * @return neuron weights; size of the array is equal to inputSize + 1
	 * (inputSize is specified in the constructor {@link #LastNeuron(int, double[], double)});
	 * first value represents bias unit weight
	 */
	public double getExpectedValue(int position) {

		try {
			double res = expectedValues.get(position);
			return res;
		} catch (Exception e) {
			throw new RuntimeException(Integer.valueOf(position).toString() + " " + 
					Boolean.valueOf(expectedValues.containsKey(0)).toString() + " " +
					Boolean.valueOf(expectedValues.containsKey(1)).toString() + " " +
					Boolean.valueOf(expectedValues.containsKey(2)).toString() + " " +
					Boolean.valueOf(expectedValues.containsKey(3)).toString());
		}
	}
	
}
