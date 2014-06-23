package pl.edu.agh.suu.storm.neural;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.Date;

/**
 * This class contains data logging utils.
 */
public class ResultLogger {

	private static final String FILENAME = "/home/hduser/nnres.log";
	
	/**
	 * Logs result of an iteration to the file.
	 * @param iteration number of iteration
	 * @param error computed error
	 * @param foundValid number of algorithm valid matches
	 */
	public static void logResult(int iteration, double error, int foundValid) {
		try {
			DateFormat dateFormat = new SimpleDateFormat("HH:mm:ss:SSS");
			Date date = new Date();
			StringBuilder sb = new StringBuilder();
			sb.append(dateFormat.format(date)).append(" | ").append("iteration: ").append(iteration).append(", ").append("error: ")
					.append(error).append(", ").append("foundValid: ").append(foundValid).append("\n");
			FileWriter fileWriter = new FileWriter(FILENAME, true);
			BufferedWriter bufferedWriter = new BufferedWriter(fileWriter);
			bufferedWriter.write(sb.toString());
			bufferedWriter.close();
		} catch (IOException e) {
			throw new RuntimeException(e);
		}
	}
	
}
