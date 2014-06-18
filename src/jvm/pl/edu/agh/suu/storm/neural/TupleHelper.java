package pl.edu.agh.suu.storm.neural;

import backtype.storm.tuple.Tuple;

/**
 * This class contains methods for tuple manipulation.
 */
public class TupleHelper {

	private TupleHelper() {
	}
	
	public static final String FORWARD = "forward";
	public static final String BACKWARD = "backward";
	public static final String DATA = "data";
	public static final String ITERATION_START = "iteration_start";
	public static final String ITERATION_END = "iteration_end";
	public static final String GLOBAL_BEGIN = "global_begin";
	
	/**
	 * Returns type of the tuple.
	 * @param tuple input tuple
	 * @return Type of the tuple equal to one of the string constants from this class.
	 */
	public static String getType(Tuple tuple) {
		return tuple.getString(0);
	}
	
	/**
	 * Returns size of the layer which has sent the tuple.
	 * @param tuple input tuple
	 * @return size of the layer which has sent the tuple
	 */
	public static int getLayerSize(Tuple tuple) {
		return tuple.getInteger(1);
	}
	
	/**
	 * Returns id of the layer which has sent the tuple.
	 * @param tuple input tuple
	 * @return id of the layer which has sent the tuple
	 */
	public static int getLayerId(Tuple tuple) {
		return tuple.getInteger(2);
	}
	
	/**
	 * Returns position of the element in the training set.
	 * Only valid for {@link #FORWARD} and {@link #BACKWARD} types.
	 * @param tuple input tuple
	 * @return position of the element in the training set
	 */
	public static int getElementId(Tuple tuple) {
		return tuple.getInteger(3);
	}
	
	/**
	 * Returns data for the forward propagation.
	 * Only valid for {@link #FORWARD} type.
	 * @param tuple input tuple
	 * @return data for the forward propagation (input values from the previous layer)
	 */
	public static double[] getForwardData(Tuple tuple) {
		return (double[]) tuple.getValue(4);
	}
	
	/**
	 * Returns data for the backward propagation.
	 * Only valid for {@link #BACKWARD} type.
	 * @param tuple input tuple
	 * @return data for the backward propagation (input values from the next layer)
	 */
	public static double[][] getBackwardData(Tuple tuple) {
		return (double[][]) tuple.getValue(4);
	}
	
	/**
	 * Returns data for the iteration end.
	 * Only valid for {@link #ITERATION_END} type.
	 * @param tuple input tuple
	 * @return data for the iteration end (training set size)
	 */
	public static int getIterationEndData(Tuple tuple) {
		return tuple.getInteger(4);
	}
	
	/**
	 * Returns position of the element in the training set.
	 * Only valid for {@link #DATA} type.
	 * @param tuple input tuple
	 * @return position of the element in the traning set
	 */
	public static int getDataKey(Tuple tuple) {
		return (int) (((double[]) tuple.getValue(4))[0]);
	}
	
	/**
	 * Returns expected output value of the neural network.
	 * Only valid for {@link #DATA} type.
	 * @param tuple input tuple
	 * @return expected output value of the neural network
	 */
	public static double getDataValue(Tuple tuple) {
		return ((double[]) tuple.getValue(4))[1];
	}
	
}
