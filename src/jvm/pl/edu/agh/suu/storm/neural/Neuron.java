package pl.edu.agh.suu.storm.neural;

import java.io.Serializable;

public class Neuron implements Serializable {

	private static final long serialVersionUID = 6170878261369297308L;
	
	private int inputSize;
	private double[] weights; // length of weights is inputSize + 1 (bias)
	private double[] forwardInput; // length of forwardInput is inputSize (no bias)
	private double alpha; // convergence coefficient
	private double delta = 0.0d;
	private double[] derivatives;
	
	public Neuron(int inputSize, double[] weights, double alpha) {
		this.inputSize = inputSize;
		this.weights = weights;
		this.alpha = alpha;
	}
	
	public double propagateForward(double[] input) { // input - neurons from previous layer
		this.forwardInput = input; // saved for computing derivative in backward propagation later
		double sum = weights[0]; // bias
		for (int i = 0; i < inputSize; ++i) {
			sum += input[i] * weights[i + 1];
		}
		return Functions.sigmoid(sum);
	}
	
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
	
	public void updateWeights(int trainingSetSize) {
		weights[0] = weights[0] - alpha * (delta / trainingSetSize); // bias
		for (int i = 0; i < inputSize; ++i) {
			weights[i + 1] = weights[i + 1] - alpha * (derivatives[i] / trainingSetSize);
		}
		resetDelta();
	}
	
	public void alterAlpha(double newAlpha) {
		this.alpha = newAlpha;
	}
	
	public void resetDelta() {
		this.delta = 0.0d;
		this.derivatives = new double[inputSize];
		for (int i = 0; i < inputSize; ++i) {
			this.derivatives[i] = 0.0d;
		}
	}
	
	public double[] getWeights() {
		return weights;
	}
	
}
