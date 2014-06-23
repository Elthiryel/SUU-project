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
	
	private int iter = 0;
	
	@SuppressWarnings("rawtypes")
	@Override
	public void open(Map conf, TopologyContext context, SpoutOutputCollector collector) {
		_collector = collector;
	}
	
	@Override
	public void nextTuple() {
		if (iter < 1) {
			_collector.emit(new Values(TupleHelper.GLOBAL_BEGIN, 0, 0, 0, 0));
			++iter;
		}
	}
	
	@Override
	public void declareOutputFields(OutputFieldsDeclarer declarer) {
		declarer.declare(new Fields("type", "layerSize", "layerId", "elementId", "data"));
	}
	
}
