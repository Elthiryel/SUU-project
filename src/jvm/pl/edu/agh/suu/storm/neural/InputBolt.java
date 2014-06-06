package pl.edu.agh.suu.storm.neural;

import java.util.Map;

import backtype.storm.task.OutputCollector;
import backtype.storm.task.TopologyContext;
import backtype.storm.topology.OutputFieldsDeclarer;
import backtype.storm.topology.base.BaseRichBolt;
import backtype.storm.tuple.Fields;
import backtype.storm.tuple.Tuple;
import backtype.storm.tuple.Values;

public class InputBolt extends BaseRichBolt {

	private static final long serialVersionUID = -3473671857504101756L;
	
	private OutputCollector _collector;
	private int valuesIter = 0;
	
	private double[] input0 = { 0.0d, 0.0d };
	private double[] input1 = { 0.0d, 1.0d };
	private double[] input2 = { 1.0d, 0.0d };
	private double[] input3 = { 1.0d, 1.0d };
	
	private Values[] values = {
			new Values(TupleHelper.FORWARD, 2, 0, 0, input0),
			new Values(TupleHelper.FORWARD, 2, 0, 1, input1),
			new Values(TupleHelper.FORWARD, 2, 0, 2, input2),
			new Values(TupleHelper.FORWARD, 2, 0, 3, input3),
	};
	
	@SuppressWarnings("rawtypes")
	@Override
	public void prepare(Map stormConf, TopologyContext context, OutputCollector collector) {
		_collector = collector;
	}
	
	@Override
	public void execute(Tuple tuple) {
		String type = TupleHelper.getType(tuple);
		if (type.equals(TupleHelper.GLOBAL_BEGIN)) {
			_collector.emit(new Values(TupleHelper.ITERATION_START, 2, 0, 0, 0));
			valuesIter = 0;
			_collector.emit(values[valuesIter]);
			++valuesIter;
		} else if (type.equals(TupleHelper.BACKWARD)) {
			if (valuesIter < 4) {
				_collector.emit(values[valuesIter]);
				++valuesIter;
			} else {
				_collector.emit(new Values(TupleHelper.ITERATION_END, 2, 0, 0, 4));
				valuesIter = 0;
				_collector.emit(new Values(TupleHelper.ITERATION_START, 2, 0, 0, 0));
				_collector.emit(values[valuesIter]);
				++valuesIter;
			}
		}
	}
	
	@Override
	public void declareOutputFields(OutputFieldsDeclarer declarer) {
		declarer.declare(new Fields("type", "layerSize", "layerId", "elementId", "data"));
	}
	
}
