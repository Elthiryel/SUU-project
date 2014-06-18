package pl.edu.agh.suu.storm.neural;

import java.io.Serializable;

/**
 * This class represents a hidden layer of the neural network.
 */
public class Layer implements Serializable {

	private static final long serialVersionUID = -953475543920461902L;
	
	private Neuron[] neurons;
	private int layerSize;
	
	/**
	 * Creates a new hidden layer.
	 * @param inputSize size of the previous layer
	 * @param layerSize size of this layer (number of neurons)
	 * @param alpha neural network learning rate
	 */
	public Layer(int inputSize, int layerSize, double alpha) {
		this.layerSize = layerSize;
		neurons = new Neuron[layerSize];
		for (int i = 0; i < layerSize; ++i) {
			double weights[] = new double[inputSize + 1];
			for (int j = 0; j < inputSize + 1; ++j) {
				weights[j] = Functions.randomWeight();
			}
			neurons[i] = new Neuron(inputSize, weights, alpha);
		}
	}
	
	/**
	 * Performs the forward propagation (output value computation).
	 * @param input input from the previous layer; size of the array should be equal to inputSize specified in the constructor
	 * 	      ({@link #Layer(int, int, double)})
	 * @return computed output values for all neurons in this layer; size of the array is equal to layerSize specified in the
	 *        constructor ({@link #Layer(int, int, double)})
	 */
	public double[] propagateForward(double[] input) { // input - neurons from previous layer
		double[] results = new double[layerSize];
		for (int i = 0; i < layerSize; ++i) {
			results[i] = neurons[i].propagateForward(input);
		}
		return results;
	}
	
	/**
	 * Performs the backward propagation. This method does not perform the neuron weights update itself.
	 * After passing all training data through forward and backward propagation (when iteration is over)
	 * one should call {@link #updateWeights(int)} to recalculate neuron weights.
	 * @param nextLayerValues backward propagation result from the next layer
	 * @param nextLayerSize size of the {@link nextLayerValues}
	 * @return backward propagation result; size of the array is equal to layerSize x inputSize (specified in the constructor
	 * 	       {@link #Neuron(int, double[], double)})
	 */
	public double[][] propagateBackward(double[][] nextLayerValues, int nextLayerSize) {
		double[][] results = new double[layerSize][];
		for (int i = 0; i < layerSize; ++i) {
			double[] neuronInput = new double[nextLayerSize];
			for (int j = 0; j < nextLayerSize; ++j) {
				neuronInput[j] = nextLayerValues[j][i];
			}
			results[i] = neurons[i].propagateBackward(neuronInput, nextLayerSize);
		}
		return results;
	}
	
	/**
	 * Updates the neuron weights after finishing the iteration.
	 * @param trainingSetSize training set element count
	 * @see #propagateBackward(double[][], int)
	 */
	public void updateWeights(int trainingSetSize) {
		for (Neuron n : neurons) {
			n.updateWeights(trainingSetSize);
		}
	}
	
	/**
	 * Alters neural network learning rate.
	 * @param newAlpha new neural network learning rate
	 */
	public void alterAlpha(double newAlpha) {
		for (Neuron n : neurons) {
			n.alterAlpha(newAlpha);
		}
	}
	
	/**
	 * Should be called at the beginning of an iteration to perform necessary cleanup.
	 */
	public void resetDelta() {
		for (Neuron n : neurons) {
			n.resetDelta();
		}
	}
	
	/**
	 * Returns weights of the neurons from this layer.
	 * @return neuron weights; size of the array is equal to layerSize x (inputSize + 1) (specified in the constructor
	 *         {@link #Layer(int, int, double)}); first value in each row represents bias unit weight
	 */
	public double[][] getWeights() {
		double[][] result = new double[layerSize][];
		for (int i = 0; i < layerSize; ++i) {
			result[i] = neurons[i].getWeights();
		}
		return result;
	}
	
}
