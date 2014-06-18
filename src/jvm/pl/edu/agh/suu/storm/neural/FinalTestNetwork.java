package pl.edu.agh.suu.storm.neural;

import backtype.storm.Config;
import backtype.storm.LocalCluster;
import backtype.storm.StormSubmitter;
import backtype.storm.generated.AlreadyAliveException;
import backtype.storm.generated.InvalidTopologyException;
import backtype.storm.topology.BoltDeclarer;
import backtype.storm.topology.TopologyBuilder;
import backtype.storm.utils.Utils;

public class FinalTestNetwork {

	private static double ALPHA = 0.03d;
	
	public static void main(String[] args) {
		
		TopologyBuilder builder = new TopologyBuilder();
		
		builder.setSpout("input", new InputSpout(), 1);
		BoltDeclarer boltInput = builder.setBolt("boltInput", new InputBolt(), 1);
		BoltDeclarer layer1 = builder.setBolt("layer1", new LayerBolt(2, 2, ALPHA, 1), 1);
		double[] startWeights = { Functions.randomWeight(), Functions.randomWeight(), Functions.randomWeight() };
		BoltDeclarer output = builder.setBolt("output", new LastNeuronBolt(2, startWeights, ALPHA, 2), 1);
		
		boltInput.shuffleGrouping("input").shuffleGrouping("layer1");
		layer1.shuffleGrouping("boltInput").shuffleGrouping("output");
		output.shuffleGrouping("layer1").shuffleGrouping("input");
		
		Config conf = new Config();
		conf.setDebug(false);
		conf.setNumWorkers(8);
		
		try {
			//LocalCluster cluster = new LocalCluster();
			//cluster.submitTopology("myNeuralNetwork", conf, builder.createTopology());
			StormSubmitter.submitTopology("myNeuralNetwork", conf, builder.createTopology());
			//Utils.sleep(100000);
			//cluster.killTopology("test");
			//cluster.shutdown();
		} catch (InvalidTopologyException e) {
			e.printStackTrace();
		} catch (AlreadyAliveException e) {
			e.printStackTrace();
		}
		
	}
}
