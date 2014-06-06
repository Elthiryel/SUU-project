package pl.edu.agh.suu.storm.neural;

import java.util.Random;

public class Functions {

	private static Random random = new Random();
	
	public static double sigmoid(double input) {
		double up = 1.0d;
		double down = 1.0d + Math.exp(-1.0d * input);
		return up / down;
	}
	
	public static double randomWeight() {
		return random.nextDouble();
	}
	
	public static void normalize(double[][] input, int trainingSetSize, int numberOfFeatures) {
		double[] averages = new double[numberOfFeatures];
		double[] squareAverages = new double[numberOfFeatures];
		for (int i = 0; i < numberOfFeatures; ++i) {
			averages[i] = 0.0d;
			squareAverages[i] = 0.0d;
		}
		for (int i = 0; i < trainingSetSize; ++i) {
			for (int j = 0; j < numberOfFeatures; ++j) {
				averages[j] += input[i][j] / trainingSetSize;
				squareAverages[j] += input[i][j] * input[i][j] / trainingSetSize;
			}
		}
		for (int i = 0; i < trainingSetSize; ++i) {
			for (int j = 0; j < numberOfFeatures; ++j) {
				input[i][j] = (input[i][j] - averages[j]) - Math.sqrt(squareAverages[j] - averages[j] * averages[j]);
			}
		}
	}
	
}
