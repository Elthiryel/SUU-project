package pl.edu.agh.suu.storm.neural;

import java.util.Map;

import backtype.storm.task.OutputCollector;
import backtype.storm.task.TopologyContext;
import backtype.storm.topology.OutputFieldsDeclarer;
import backtype.storm.topology.base.BaseRichBolt;
import backtype.storm.tuple.Fields;
import backtype.storm.tuple.Tuple;
import backtype.storm.tuple.Values;

/**
 * This class is an Apache Storm bolt for handling neural network output layer with single neuron.
 */
public class LastNeuronBolt extends BaseRichBolt {

	private static final long serialVersionUID = -7529513948694806382L;
	
	private OutputCollector _collector;
	private LastNeuron neuron;
	private int layerId;
	private int dataSize;
	private double error;
	private int foundValid;
	private int iteration;
	
	/**
	 * Creates new bolt for handling single neuron output layer.
	 * @param inputSize size of the previous layer
	 * @param weights initial neuron weights;
	 *        size of the array should be equal to inputSize + 1; first value represent bias unit weight
	 * @param alpha neural network learning rate
	 * @param layerId number of the layer in the neural network
	 */
	public LastNeuronBolt(int inputSize, double weights[], double alpha, int layerId, int dataSize) {
		this.neuron = new LastNeuron(inputSize, weights, alpha);
		this.layerId = layerId;
		this.dataSize = dataSize;
		this.iteration = 0;
	}
	
	@SuppressWarnings("rawtypes")
	@Override
	public void prepare(Map stormConf, TopologyContext context, OutputCollector collector) {
		_collector = collector;
	}
	
	@Override
	public void execute(Tuple tuple) {
		String type = TupleHelper.getType(tuple);
		int otherLayerId = TupleHelper.getLayerId(tuple);
		int elementId = TupleHelper.getElementId(tuple);
		if (type.equals(TupleHelper.FORWARD) && otherLayerId + 1 == layerId) {
			double result = neuron.propagateForward(TupleHelper.getForwardData(tuple));
			Double expectedValue = null;
			while (expectedValue == null) {
				try {
					expectedValue = neuron.getExpectedValue(elementId);
				} catch (Exception e) {
					System.out.println("Value not there yet.");
				}
			}
			if (iteration % 20 == 0 && elementId < 20)
				System.out.println("LAST_NEURON_BOLT | element: " + elementId + ", result: " + result + ", expected: " + expectedValue);
			double cost = (-expectedValue * Math.log(result)) - ((1 - expectedValue) * (Math.log(1 - result)));
			error += cost;
			if (Math.abs(result - expectedValue) < 0.5d) {
				++foundValid;
			}
			double[] propagationResult = neuron.propagateBackward(elementId);
			double[][] toSend = new double[1][];
			toSend[0] = propagationResult;
			_collector.emit(new Values(TupleHelper.BACKWARD, 1, layerId, elementId, toSend));
		} else if (type.equals(TupleHelper.ITERATION_START) && otherLayerId + 1 == layerId) {
			error = 0.0d;
			foundValid = 0;
			neuron.resetDelta();
		} else if (type.equals(TupleHelper.ITERATION_END) && otherLayerId + 1 == layerId) {
			int trainingSetSize = TupleHelper.getIterationEndData(tuple);
			error /= trainingSetSize;
			//System.out.println("LAST_NEURON_BOLT | iteration: " + iteration + ", error: " + error + ", valid: " + foundValid);
			ResultLogger.logResult(iteration, error, foundValid);
			++iteration;
			neuron.updateWeights(trainingSetSize);
		} else if (type.equals(TupleHelper.DATA)) {
			int key = TupleHelper.getDataKey(tuple);
			double value = TupleHelper.getDataValue(tuple);
			neuron.setExpectedValue(value, key);
			if (key + 1 == this.dataSize) {
				_collector.emit(new Values(TupleHelper.DATA_TRANSFERRED, 0, 0, 0, 0));
			}
		}
		_collector.ack(tuple);
	}
	
	@Override
	public void declareOutputFields(OutputFieldsDeclarer declarer) {
		declarer.declare(new Fields("type", "layerSize", "layerId", "elementId", "data"));
	}
	
}
