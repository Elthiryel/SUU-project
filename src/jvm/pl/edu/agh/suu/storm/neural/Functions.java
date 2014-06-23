package pl.edu.agh.suu.storm.neural;

import java.util.Random;

/**
 * This class contains utility methods.
 */
public class Functions {

	private static Random random = new Random();
	
	private Functions() {
	}
	
	/**
	 * Calculates the sigmoid function value for the specified input.
	 * @param input input to the sigmoid function
	 * @return sigmoid function result
	 */
	public static double sigmoid(double input) {
		double up = 1.0d;
		double down = 1.0d + Math.exp(-1.0d * input);
		return up / down;
	}
	
	/**
	 * Returns random neuron weight.
	 * @return random neuron weight
	 */
	public static double randomWeight() {
		return 2 * (random.nextDouble() - 0.5);
	}
	
	/**
	 * Performs normalization over the set of values.
	 * @param input data to normalize; size of the array should be equal to {@link trainingSetSize} x {@link numberOfFeatures}
	 * @param trainingSetSize size of the data set
	 * @param numberOfFeatures number of features
	 */
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
	
	/**
	 * Generates random data for classification with four features and expected output (0 or 1).
	 * @param from initial element id (inclusive)
	 * @param to final element id (exclusive)
	 * @param seed random seed
	 * @return generated data
	 */
	public static double[][] generateRandomData(int from, int to, long seed) {
		double[][] toReturn = new double[to - from][];
		for (int i = from; i < to; ++i) {
			Random random = new Random(seed * i);
			double[] row = new double[5];
			for (int j = 0; j < 4; ++j) {
				row[j] = random.nextDouble();
			}
			double decisionFactor = 3 * (row[0] * row[0]) + (Math.exp(row[1])) - 2 * (row[2] * row[2] * row[2]) + 5 * Math.sqrt(row[3]);
			if (decisionFactor > 4.86d) {
				row[4] = 1.0d;
			} else {
				row[4] = 0.0d;
			}
			toReturn[i - from] = row;
		}
		return toReturn;
	}
	
}
