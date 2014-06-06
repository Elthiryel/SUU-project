package pl.edu.agh.suu.storm.neural;

import backtype.storm.tuple.Tuple;

public class TupleHelper {

	public static final String FORWARD = "forward";
	public static final String BACKWARD = "backward";
	public static final String DATA = "data";
	public static final String ITERATION_START = "iteration_start";
	public static final String ITERATION_END = "iteration_end";
	public static final String GLOBAL_BEGIN = "global_begin";
	
	public static String getType(Tuple tuple) {
		return tuple.getString(0);
	}
	
	public static int getLayerSize(Tuple tuple) {
		return tuple.getInteger(1);
	}
	
	public static int getLayerId(Tuple tuple) {
		return tuple.getInteger(2);
	}
	
	public static int getElementId(Tuple tuple) {
		return tuple.getInteger(3);
	}
	
	public static double[] getForwardData(Tuple tuple) {
		return (double[]) tuple.getValue(4);
	}
	
	public static double[][] getBackwardData(Tuple tuple) {
		return (double[][]) tuple.getValue(4);
	}
	
	public static int getIterationEndData(Tuple tuple) {
		return tuple.getInteger(4);
	}
	
	public static int getDataKey(Tuple tuple) {
		return (int) (((double[]) tuple.getValue(4))[0]);
	}
	
	public static double getDataValue(Tuple tuple) {
		return ((double[]) tuple.getValue(4))[1];
	}
	
}
