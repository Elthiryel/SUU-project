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
 * This class is an Apache Storm bolt for handling neural network hidden layer.
 */
public class LayerBolt extends BaseRichBolt {

	private static final long serialVersionUID = -7165958052675311644L;
	
	private OutputCollector _collector;
	private Layer layer;
	private int layerSize;
	private int layerId;
	
	/**
	 * Creates a new bolt for handling hidden layer.
	 * @param inputSize size of the previous layer
	 * @param layerSize size of this layer (number of neurons)
	 * @param alpha neural network learning rate
	 * @param layerId number of the layer in the neural network
	 */
	public LayerBolt(int inputSize, int layerSize, double alpha, int layerId) {
		this.layer = new Layer(inputSize, layerSize, alpha);
		this.layerSize = layerSize;
		this.layerId = layerId;
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
			double[] propagationResult = layer.propagateForward(TupleHelper.getForwardData(tuple));
			_collector.emit(new Values(TupleHelper.FORWARD, layerSize, layerId, elementId, propagationResult));
		} else if (type.equals(TupleHelper.BACKWARD) && otherLayerId - 1 == layerId) {
			double[][] propagationResult = layer.propagateBackward(TupleHelper.getBackwardData(tuple), TupleHelper.getLayerSize(tuple));
			_collector.emit(new Values(TupleHelper.BACKWARD, layerSize, layerId, elementId, propagationResult));
		} else if (type.equals(TupleHelper.ITERATION_START) && otherLayerId + 1 == layerId) {
			layer.resetDelta();
			_collector.emit(new Values(TupleHelper.ITERATION_START, 0, layerId, elementId, 0));
		} else if (type.equals(TupleHelper.ITERATION_END) && otherLayerId + 1 == layerId) {
			int trainingSetSize = TupleHelper.getIterationEndData(tuple);
			layer.updateWeights(trainingSetSize);
			_collector.emit(new Values(TupleHelper.ITERATION_END, 0, layerId, elementId, trainingSetSize));
		}
		_collector.ack(tuple);
	}
	
	@Override
	public void declareOutputFields(OutputFieldsDeclarer declarer) {
		declarer.declare(new Fields("type", "layerSize", "layerId", "elementId", "data"));
	}
	
}
