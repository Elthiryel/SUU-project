package pl.edu.agh.suu.storm.neural;

import java.io.Serializable;
import java.util.HashMap;
import java.util.Map;

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
	
	public LastNeuron(int inputSize, double[] weights, double alpha) {
		this.inputSize = inputSize;
		this.weights = weights;
		this.alpha = alpha;
		this.expectedValues = new HashMap<Integer, Double>();
	}
	
	public double propagateForward(double[] input) { // input - neurons from previous layer
		this.forwardInput = input; // saved for computing derivative in backward propagation later
		double sum = weights[0]; // bias
		for (int i = 0; i < inputSize; ++i) {
			sum += input[i] * weights[i + 1];
		}
		result = Functions.sigmoid(sum);
		return result;
	}
	
	public double[] propagateBackward(int numberOfElement) {
		double delta = result - expectedValues.get(numberOfElement);
		double[] returnValues = new double[inputSize];
		for (int i = 0; i < inputSize; ++i) {
			returnValues[i] = weights[i + 1] * delta * forwardInput[i] * (1 - forwardInput[i]);
			derivatives[i] += forwardInput[i] * delta;
		}
		this.delta += delta;
		//System.out.println("TEMP DELTA: " + delta + ", TOTAL DELTA: " + this.delta);
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
	
	public void setExpectedValue(double value, int position) { // allows streaming
		this.expectedValues.put(position, value);
	}
	
	public void setExpectedValues(double[] values, int count) {
		for (int i = 0; i < count; ++i) {
			this.expectedValues.put(i, values[i]);
		}
	}
	
	public void setExpectedValues(int[] keys, double[] values, int count) {
		for (int i = 0; i < count; ++i) {
			this.expectedValues.put(keys[i], values[i]);
		}
	}
	
	public void cleanExpectedValue(int position) {
		this.expectedValues.remove(position);
	}
	
	public double[] getWeights() {
		return weights;
	}
	
	public double getExpectedValue(int position) {
		return expectedValues.get(position);
	}
	
}
