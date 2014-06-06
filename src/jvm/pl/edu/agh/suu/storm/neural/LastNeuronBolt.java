package pl.edu.agh.suu.storm.neural;

import java.util.Map;

import backtype.storm.task.OutputCollector;
import backtype.storm.task.TopologyContext;
import backtype.storm.topology.OutputFieldsDeclarer;
import backtype.storm.topology.base.BaseRichBolt;
import backtype.storm.tuple.Fields;
import backtype.storm.tuple.Tuple;
import backtype.storm.tuple.Values;

public class LastNeuronBolt extends BaseRichBolt {

	private static final long serialVersionUID = -7529513948694806382L;
	
	private OutputCollector _collector;
	private LastNeuron neuron;
	private int layerId;
	
	public LastNeuronBolt(int inputSize, double weights[], double alpha, int layerId) {
		this.neuron = new LastNeuron(inputSize, weights, alpha);
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
			double result = neuron.propagateForward(TupleHelper.getForwardData(tuple));
			System.out.println("LAST_NEURON_BOLT | element: " + elementId + ", result: " + result + ", expected: " + neuron.getExpectedValue(elementId));
			double[] propagationResult = neuron.propagateBackward(elementId);
			double[][] toSend = new double[1][];
			toSend[0] = propagationResult;
			_collector.emit(new Values(TupleHelper.BACKWARD, 1, layerId, elementId, toSend));
		} else if (type.equals(TupleHelper.ITERATION_START) && otherLayerId + 1 == layerId) {
			neuron.resetDelta();
		} else if (type.equals(TupleHelper.ITERATION_END) && otherLayerId + 1 == layerId) {
			int trainingSetSize = TupleHelper.getIterationEndData(tuple);
			neuron.updateWeights(trainingSetSize);
		} else if (type.equals(TupleHelper.DATA)) {
			int key = TupleHelper.getDataKey(tuple);
			double value = TupleHelper.getDataValue(tuple);
			neuron.setExpectedValue(value, key);
		}
	}
	
	@Override
	public void declareOutputFields(OutputFieldsDeclarer declarer) {
		declarer.declare(new Fields("type", "layerSize", "layerId", "elementId", "data"));
	}
	
}
