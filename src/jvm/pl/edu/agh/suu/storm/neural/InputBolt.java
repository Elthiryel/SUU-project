package pl.edu.agh.suu.storm.neural;

import java.util.Map;
import java.util.Random;

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
	private int globalIter = 0;
	private int valuesIter = 0;
	private int dataSize;
	private int seed;
	
	private double[][] data;
	
	public InputBolt(int dataSize) {
		this.dataSize = dataSize;
		this.seed = (new Random()).nextInt();
	}
	
	@SuppressWarnings("rawtypes")
	@Override
	public void prepare(Map stormConf, TopologyContext context, OutputCollector collector) {
		_collector = collector;
	}
	
	private void emitData() {
		int from = globalIter * dataSize;
		int to = from + dataSize;
		globalIter = (globalIter + 1) % 10;
		data = Functions.generateRandomData(from, to, seed);
		for (int i = 0; i < dataSize; ++i) {
			double[] array = { i, data[i][4] };
			_collector.emit(new Values(TupleHelper.DATA, 4, 0, i, array));
		}
	}
	
	@Override
	public void execute(Tuple tuple) {
		String type = TupleHelper.getType(tuple);
		if (type.equals(TupleHelper.GLOBAL_BEGIN)) {
			emitData();
		} else if (type.equals(TupleHelper.BACKWARD) && TupleHelper.getLayerId(tuple) == 1) {
			if (valuesIter < dataSize) {
				_collector.emit(new Values(TupleHelper.FORWARD, 4, 0, valuesIter, data[valuesIter]));
				++valuesIter;
			} else {
				_collector.emit(new Values(TupleHelper.ITERATION_END, 4, 0, 0, dataSize));
				emitData();
			}
		} else if (type.equals(TupleHelper.DATA_TRANSFERRED)) {
			_collector.emit(new Values(TupleHelper.ITERATION_START, 4, 0, 0, 0));
			valuesIter = 0;
			_collector.emit(new Values(TupleHelper.FORWARD, 4, 0, valuesIter, data[valuesIter]));
			++valuesIter;
		}
		_collector.ack(tuple);
	}
	
	@Override
	public void declareOutputFields(OutputFieldsDeclarer declarer) {
		declarer.declare(new Fields("type", "layerSize", "layerId", "elementId", "data"));
	}
	
}
