package pl.edu.agh.suu.storm.neural;

import java.util.Map;

import backtype.storm.spout.SpoutOutputCollector;
import backtype.storm.task.TopologyContext;
import backtype.storm.topology.OutputFieldsDeclarer;
import backtype.storm.topology.base.BaseRichSpout;
import backtype.storm.tuple.Fields;
import backtype.storm.tuple.Values;

public class InputSpout extends BaseRichSpout {

	private static final long serialVersionUID = -3347473897159538942L;
	
	private SpoutOutputCollector _collector;
	
	private double[] arr0 = { 0, 0.0d };
	private double[] arr1 = { 1, 1.0d };
	private double[] arr2 = { 2, 1.0d };
	private double[] arr3 = { 3, 0.0d };
	
	private int iter = 0;
	
	private Values[] values = {
			new Values(TupleHelper.DATA, 2, 0, 0, arr0),
			new Values(TupleHelper.DATA, 2, 0, 1, arr1),
			new Values(TupleHelper.DATA, 2, 0, 2, arr2),
			new Values(TupleHelper.DATA, 2, 0, 3, arr3),
			new Values(TupleHelper.GLOBAL_BEGIN, 0, 0, 0, 0)
	};
	
	@SuppressWarnings("rawtypes")
	@Override
	public void open(Map conf, TopologyContext context, SpoutOutputCollector collector) {
		_collector = collector;
	}
	
	@Override
	public void nextTuple() {
		if (iter < 5) {
			_collector.emit(values[iter]);
			++iter;
		}
	}
	
	@Override
	public void declareOutputFields(OutputFieldsDeclarer declarer) {
		declarer.declare(new Fields("type", "layerSize", "layerId", "elementId", "data"));
	}
	
}
