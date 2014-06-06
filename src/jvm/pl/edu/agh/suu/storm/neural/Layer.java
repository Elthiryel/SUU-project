package pl.edu.agh.suu.storm.neural;

import java.io.Serializable;

public class Layer implements Serializable {

	private static final long serialVersionUID = -953475543920461902L;
	
	private Neuron[] neurons;
	private int layerSize;
	
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
	
	public double[] propagateForward(double[] input) { // input - neurons from previous layer
		double[] results = new double[layerSize];
		for (int i = 0; i < layerSize; ++i) {
			results[i] = neurons[i].propagateForward(input);
		}
		return results;
	}
	
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
	
	public void updateWeights(int trainingSetSize) {
		for (Neuron n : neurons) {
			n.updateWeights(trainingSetSize);
		}
	}
	
	public void alterAlpha(double newAlpha) {
		for (Neuron n : neurons) {
			n.alterAlpha(newAlpha);
		}
	}
	
	public void resetDelta() {
		for (Neuron n : neurons) {
			n.resetDelta();
		}
	}
	
	public double[][] getWeights() {
		double[][] result = new double[layerSize][];
		for (int i = 0; i < layerSize; ++i) {
			result[i] = neurons[i].getWeights();
		}
		return result;
	}
	
}
