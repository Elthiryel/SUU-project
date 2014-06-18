package pl.edu.agh.suu.storm.neural;

import java.io.Serializable;

/**
 * This class represents a single neuron in the hidden layer of the neural network.
 */
public class Neuron implements Serializable {

	private static final long serialVersionUID = 6170878261369297308L;
	
	private int inputSize;
	private double[] weights; // length of weights is inputSize + 1 (bias)
	private double[] forwardInput; // length of forwardInput is inputSize (no bias)
	private double alpha; // convergence coefficient
	private double delta = 0.0d;
	private double[] derivatives;
	
	/**
	 * Creates a new neuron.
	 * @param inputSize size of the previous layer
	 * @param weights initial neuron weights;
	 *        size of the array should be equal to inputSize + 1; first value represent bias unit weight
	 * @param alpha neural network learning rate
	 */
	public Neuron(int inputSize, double[] weights, double alpha) {
		this.inputSize = inputSize;
		this.weights = weights;
		this.alpha = alpha;
	}
	
	/**
	 * Performs the forward propagation (output value computation).
	 * @param input input from the previous layer; size of the array should be equal to inputSize specified in the constructor
	 * 	      ({@link #Neuron(int, double[], double)})
	 * @return computed output value
	 */
	public double propagateForward(double[] input) { // input - neurons from previous layer
		this.forwardInput = input; // saved for computing derivative in backward propagation later
		double sum = weights[0]; // bias
		for (int i = 0; i < inputSize; ++i) {
			sum += input[i] * weights[i + 1];
		}
		return Functions.sigmoid(sum);
	}
	
	/**
	 * Performs the backward propagation. This method does not perform the neuron weights update itself.
	 * After passing all training data through forward and backward propagation (when iteration is over)
	 * one should call {@link #updateWeights(int)} to recalculate neuron weights.
	 * @param nextLayerValues backward propagation result from the next layer
	 * @param nextLayerSize size of the {@link nextLayerValues}
	 * @return backward propagation result; size of the array is equal to inputSize specified in the constructor
	 * 	       ({@link #Neuron(int, double[], double)})
	 */
	public double[] propagateBackward(double[] nextLayerValues, int nextLayerSize) {
		double delta = 0.0d;
		for (int i = 0; i < nextLayerSize; ++i) {
			delta += nextLayerValues[i];
		}
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
	 * @see #propagateBackward(double[], int)
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
	 * Returns neuron weights.
	 * @return neuron weights; size of the array is equal to inputSize + 1
	 * (inputSize is specified in the constructor {@link #Neuron(int, double[], double)});
	 * first value represents bias unit weight
	 */
	public double[] getWeights() {
		return weights;
	}
	
}
