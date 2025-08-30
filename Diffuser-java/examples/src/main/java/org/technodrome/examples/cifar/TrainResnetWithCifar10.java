/*
 * Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file except in compliance
 * with the License. A copy of the License is located at
 *
 * http://aws.amazon.com/apache2.0/
 *
 * or in the "license" file accompanying this file. This file is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES
 * OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions
 * and limitations under the License.
 */
package org.technodrome.examples.cifar;

import ai.djl.Device;
import ai.djl.Model;
import ai.djl.ModelException;
import ai.djl.basicdataset.cv.classification.Cifar10;
import ai.djl.basicmodelzoo.cv.classification.ResNetV1;
import ai.djl.inference.Predictor;
import ai.djl.metric.Metrics;
import ai.djl.modality.Classifications;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.modality.cv.transform.Normalize;
import ai.djl.modality.cv.transform.ToTensor;
import ai.djl.modality.cv.translator.ImageClassificationTranslator;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Block;
import ai.djl.nn.BlockFactory;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.EasyTrain;
import ai.djl.training.Trainer;
import ai.djl.training.TrainingResult;
import ai.djl.training.dataset.Dataset;
import ai.djl.training.dataset.RandomAccessDataset;
import ai.djl.training.evaluator.Accuracy;
import ai.djl.training.listener.TrainingListener;
import ai.djl.training.loss.Loss;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.Pipeline;
import ai.djl.translate.TranslateException;
import org.kohsuke.args4j.CmdLineException;
import org.kohsuke.args4j.CmdLineParser;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;

/**
 * An example of training an image classification (ResNet for Cifar10) model.
 *
 * <p>See this <a
 * href="https://github.com/deepjavalibrary/djl/blob/master/examples/docs/train_cifar10_resnet.md">doc</a>
 * for information about this example.
 */
public final class TrainResnetWithCifar10 {

    private static final Logger logger = LoggerFactory.getLogger(TrainResnetWithCifar10.class);

    private TrainResnetWithCifar10() {
    }

    public static void main(String[] args) throws ModelException, IOException, TranslateException {
        TrainResnetWithCifar10.runExample(args);
    }

    public static void runExample(String[] args)
            throws IOException, ModelException, TranslateException {
        CommandLineValues arguments = new CommandLineValues(args);
        CmdLineParser parser = new CmdLineParser(arguments);

        try {
            parser.parseArgument(args);
        } catch (CmdLineException e) {
            System.exit(1);
        }

        try (Model model = getModel(arguments)) {
            // get training dataset
            RandomAccessDataset trainDataset = getDataset(Dataset.Usage.TRAIN, arguments);
            RandomAccessDataset validationDataset = getDataset(Dataset.Usage.TEST, arguments);

            // setup training configuration
            DefaultTrainingConfig config = setupTrainingConfig(arguments);

            try (Trainer trainer = model.newTrainer(config)) {
                trainer.setMetrics(new Metrics());

                /*
                 * CIFAR10 is 32x32 image and pre processed into NCHW NDArray.
                 * 1st axis is batch axis, we can use 1 for initialization.
                 */
                Shape inputShape = new Shape(1, 3, 32, 32);

                // initialize trainer with proper input shape
                trainer.initialize(inputShape);
                EasyTrain.fit(trainer, arguments.getEpoch(), trainDataset, validationDataset);

                TrainingResult result = trainer.getTrainingResult();
                model.setProperty("Epoch", String.valueOf(result.getEpoch()));
                model.setProperty(
                        "Accuracy",
                        String.format("%.5f", result.getValidateEvaluation("Accuracy")));
                model.setProperty("Loss", String.format("%.5f", result.getValidateLoss()));

                Path modelPath = Paths.get("build/model");
                model.save(modelPath, "resnetv1");

                Classifications classifications = testSaveParameters(modelPath, arguments);
                logger.info("Predict result: {}", classifications.topK(3));
            }
        }
    }

    private static Model getModel(CommandLineValues arguments) throws IOException, ModelException {
        boolean preTrained = arguments.isPreTrained();
        Criteria.Builder<Image, Classifications> builder =
                Criteria.builder()
                        .setTypes(Image.class, Classifications.class)
                        .optEngine(arguments.getEngine())
                        .optProgress(new ProgressBar());
        // imperative resnet50
        if (preTrained) {
            builder.optModelUrls("djl://ai.djl.zoo/resnet/0.0.2/resnetv1");
            // load pre-trained imperative ResNet50 from DJL model zoo
            return builder.build().loadModel();
        } else {
            // construct new ResNet50 without pre-trained weights
            Model model = Model.newInstance("resnetv1", arguments.getEngine());
            Block resNet50 =
                    ResNetV1.builder()
                            .setImageShape(new Shape(3, 32, 32))
                            .setNumLayers(50)
                            .setOutSize(10)
                            .build();
            model.setBlock(resNet50);
            return model;
        }
    }

    static Classifications testSaveParameters(Path path, CommandLineValues arguments)
            throws IOException, ModelException, TranslateException {
        String synsetUrl =
                "https://mlrepo.djl.ai/model/cv/image_classification/ai/djl/mxnet/synset_cifar10.txt";
        ImageClassificationTranslator translator =
                ImageClassificationTranslator.builder()
                        .addTransform(new ToTensor())
                        .addTransform(new Normalize(Cifar10.NORMALIZE_MEAN, Cifar10.NORMALIZE_STD))
                        .optSynsetUrl(synsetUrl)
                        .optApplySoftmax(true)
                        .build();
        BlockFactory resnetFactory =
                (model, modelPath, arguments1) ->
                        ResNetV1.builder()
                                .setImageShape(new Shape(3, 32, 32))
                                .setNumLayers(50)
                                .setOutSize(10)
                                .build();

        Image img = ImageFactory.getInstance().fromFile(Paths.get("C:\\Users\\edwar\\git\\diffuser-java\\Diffuser-java\\examples\\src\\test\\resources\\airplane1.png"));

        Criteria<Image, Classifications> criteria =
                Criteria.builder()
                        .setTypes(Image.class, Classifications.class)
                        .optModelPath(path)
                        .optEngine(arguments.getEngine())
                        .optTranslator(translator)
                        .optArgument("blockFactory", resnetFactory)
                        .optModelName("resnetv1")
                        .build();

        try (ZooModel<Image, Classifications> model = criteria.loadModel();
             Predictor<Image, Classifications> predictor = model.newPredictor()) {
            return predictor.predict(img);
        }
    }

    private static DefaultTrainingConfig setupTrainingConfig(CommandLineValues arguments) {
        return new DefaultTrainingConfig(Loss.softmaxCrossEntropyLoss())
                .addEvaluator(new Accuracy())
                .optDevices(new Device[]{Device.gpu()})
                .addTrainingListeners(TrainingListener.Defaults.logging(arguments.getOutputDir()));
    }

    private static RandomAccessDataset getDataset(Dataset.Usage usage, CommandLineValues arguments)
            throws IOException {
        Pipeline pipeline =
                new Pipeline(
                        new ToTensor(),
                        new Normalize(Cifar10.NORMALIZE_MEAN, Cifar10.NORMALIZE_STD));
        Cifar10 cifar10 =
                Cifar10.builder()
                        .optUsage(usage)
                        .optManager(NDManager.newBaseManager(arguments.getEngine()))
                        .setSampling(arguments.getBatchSize(), true)
                        .optLimit(arguments.getLimit())
                        .optPipeline(pipeline)
                        .build();
        cifar10.prepare(new ProgressBar());
        return cifar10;
    }
}