package org.technodrome.examples.mnist;

import org.kohsuke.args4j.CmdLineException;
import org.kohsuke.args4j.CmdLineParser;
import org.kohsuke.args4j.Option;

public class CommandLineValues {
    @Option(name = "-e", aliases = {"--epoch"}, required = true,
            usage = "Number of epochs to train")
    private int epoch = 5;

    @Option(name = "-g", aliases = {"--max_gpus"}, required = false,
            usage = "Maximum number of gpus to use")
    private int max_gpus = 1;

    @Option(name = "-o", aliases = {"--output_dir"}, required = true,
            usage = "Output directory")
    private String outputDir = null;

    @Option(name = "-b", aliases = {"--batch_size"}, required = false,
            usage = "Batch size")
    private int batchSize = 64;

    @Option(name = "-l", aliases = {"--limit"}, required = false,
            usage = "Limit")
    private long limit = 1000L;

    private boolean errorFree = false;

    public CommandLineValues(String... args) {
        CmdLineParser parser = new CmdLineParser(this);
        parser.setUsageWidth(80);
        try {
            parser.parseArgument(args);
            errorFree = true;
        } catch (CmdLineException e) {
            System.err.println(e.getMessage());
            parser.printUsage(System.err);
        }
    }

    /**
     * Returns whether the parameters could be parsed without an
     * error.
     *
     * @return true if no error occurred.
     */
    public boolean isErrorFree() {
        return errorFree;
    }

    /**
     * Returns the source file.
     *
     * @return The source file.
     */
    public int getEpoch() {
        return epoch;
    }

    public int getMaxGpus() {
        return max_gpus;
    }

    public String getOutputDir() {
        return outputDir;
    }

    public int getBatchSize() {
        return batchSize;
    }

    public long getLimit() {
        return limit;
    }
}