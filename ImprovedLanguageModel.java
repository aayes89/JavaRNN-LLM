package javallmsimple;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.util.*;
import java.util.concurrent.*;
import java.util.stream.Collectors;

public class ImprovedLanguageModel implements Serializable {

    private static final long serialVersionUID = 1L;

    // Arquitectura mejorada
    private final Map<String, Integer> wordToIndex;
    private final Map<Integer, String> indexToWord;
    private int vocabSize;

    // Capas LSTM
    private double[][] We; // embeddings
    private double[][] Wf, Wi, Wo, Wg; // forget, input, output, candidate gates
    private double[][] Uf, Ui, Uo, Ug; // recurrent weights
    private double[] bf, bi, bo, bg; // biases
    private double[][] Why; // output weights
    private double[] by; // output bias

    // Hiperparámetros configurables
    private final int embeddingSize;
    private final int hiddenSize;
    private double learningRate;
    private final Random rand;
    private static final int MAX_VOCAB_SIZE = 50000;
    private List<String> corpusWords;

    // Mejoras para coherencia
    private final double dropoutRate = 0.2;
    private final double gradientClipNorm = 5.0;

    // Sistema de monitoreo y adaptación
    private SystemInfo systemInfo;
    private transient ExecutorService threadPool; // Marked as transient
    private int optimalBatchSize = 32;
    private int optimalThreads = 4;

    public ImprovedLanguageModel() {
        this(64, 128, 0.001);
    }

    public ImprovedLanguageModel(int embeddingSize, int hiddenSize, double learningRate) {
        this.embeddingSize = embeddingSize;
        this.hiddenSize = hiddenSize;
        this.learningRate = learningRate;
        this.wordToIndex = new HashMap<>();
        this.indexToWord = new HashMap<>();
        this.vocabSize = 0;
        this.rand = new Random(42);
        this.corpusWords = new ArrayList<>();
        this.systemInfo = new SystemInfo();
        optimizeForHardware();
    }

    public void loadCorpus(String filePath) throws IOException {
        Map<String, Integer> wordCounts = new HashMap<>();
        corpusWords.clear();

        try (BufferedReader br = new BufferedReader(
                new InputStreamReader(new FileInputStream(filePath), StandardCharsets.UTF_8))) {
            String line;
            while ((line = br.readLine()) != null) {
                String[] sentences = line.toLowerCase()
                        .replaceAll("[^a-záéíóúñü.,!?;: ]", "")
                        .split("[.!?]+");

                for (String sentence : sentences) {
                    String[] tokens = sentence.trim().split("\\s+");
                    if (tokens.length > 1) {
                        corpusWords.add("<START>");
                        for (String token : tokens) {
                            if (!token.isEmpty()) {
                                corpusWords.add(token);
                                wordCounts.put(token, wordCounts.getOrDefault(token, 0) + 1);
                            }
                        }
                        corpusWords.add("<END>");
                    }
                }
            }
        }

        wordCounts.put("<START>", 1000);
        wordCounts.put("<END>", 1000);
        wordCounts.put("<UNK>", 100);

        List<Map.Entry<String, Integer>> sortedWords = wordCounts.entrySet().stream()
                .sorted((e1, e2) -> e2.getValue().compareTo(e1.getValue()))
                .limit(MAX_VOCAB_SIZE)
                .collect(Collectors.toList());

        buildVocabulary(sortedWords);
        initializeWeights();
    }

    private void buildVocabulary(List<Map.Entry<String, Integer>> sortedWords) {
        wordToIndex.clear();
        indexToWord.clear();
        vocabSize = 0;

        for (Map.Entry<String, Integer> entry : sortedWords) {
            wordToIndex.put(entry.getKey(), vocabSize);
            indexToWord.put(vocabSize, entry.getKey());
            vocabSize++;
        }
    }

    private void initializeWeights() {
        double embScale = Math.sqrt(2.0 / embeddingSize);
        double hiddenScale = Math.sqrt(2.0 / hiddenSize);

        We = initMatrix(vocabSize, embeddingSize, embScale);

        Wf = initMatrix(hiddenSize, embeddingSize, hiddenScale);
        Wi = initMatrix(hiddenSize, embeddingSize, hiddenScale);
        Wo = initMatrix(hiddenSize, embeddingSize, hiddenScale);
        Wg = initMatrix(hiddenSize, embeddingSize, hiddenScale);

        Uf = initMatrix(hiddenSize, hiddenSize, hiddenScale);
        Ui = initMatrix(hiddenSize, hiddenSize, hiddenScale);
        Uo = initMatrix(hiddenSize, hiddenSize, hiddenScale);
        Ug = initMatrix(hiddenSize, hiddenSize, hiddenScale);

        bf = new double[hiddenSize];
        bi = new double[hiddenSize];
        bo = new double[hiddenSize];
        bg = new double[hiddenSize];
        Arrays.fill(bf, 1.0);

        Why = initMatrix(vocabSize, hiddenSize, Math.sqrt(2.0 / hiddenSize));
        by = new double[vocabSize];
    }

    private double[][] initMatrix(int rows, int cols, double scale) {
        double[][] matrix = new double[rows][cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                matrix[i][j] = rand.nextGaussian() * scale;
            }
        }
        return matrix;
    }

    private LSTMState lstmStep(double[] x, LSTMState prevState, boolean training) {
        double[] h = prevState.h;
        double[] c = prevState.c;

        double[] f = sigmoid(addVectors(addVectors(matVecMul(Wf, x), matVecMul(Uf, h)), bf));
        double[] i = sigmoid(addVectors(addVectors(matVecMul(Wi, x), matVecMul(Ui, h)), bi));
        double[] o = sigmoid(addVectors(addVectors(matVecMul(Wo, x), matVecMul(Uo, h)), bo));
        double[] g = tanh(addVectors(addVectors(matVecMul(Wg, x), matVecMul(Ug, h)), bg));

        double[] newC = addVectors(pointwiseMul(f, c), pointwiseMul(i, g));
        double[] newH = pointwiseMul(o, tanh(newC));

        if (training && dropoutRate > 0) {
            newH = applyDropout(newH, dropoutRate);
        }

        return new LSTMState(newH, newC);
    }

    public void train(int epochs, int seqLength) {
        System.out.println("=== CONFIGURACION DE ENTRENAMIENTO ===");
        systemInfo.printSystemInfo();
        System.out.println("Batch size optimo: " + optimalBatchSize);
        System.out.println("Threads: " + optimalThreads);
        System.out.println("Vocab size: " + vocabSize);
        System.out.println("Secuencias de entrenamiento: " + (corpusWords.size() / seqLength));
        System.out.println("==========================================\n");

        List<List<Integer>> sequences = prepareTrainingData(seqLength);
        System.out.println("Secuencias validas preparadas: " + sequences.size());

        for (int epoch = 0; epoch < epochs; epoch++) {
            long epochStart = System.currentTimeMillis();
            System.out.printf("\n🚀 EPOCA %d/%d\n", epoch + 1, epochs);

            double totalLoss = 0;
            int processedBatches = 0;

            Collections.shuffle(sequences, rand);

            for (int start = 0; start < sequences.size(); start += optimalBatchSize) {
                int end = Math.min(start + optimalBatchSize, sequences.size());
                List<List<Integer>> batch = sequences.subList(start, end);

                double batchLoss = processBatchParallel(batch);
                totalLoss += batchLoss;
                processedBatches++;

                if (processedBatches % 10 == 0 || processedBatches == (sequences.size() + optimalBatchSize - 1) / optimalBatchSize) {
                    double progress = (double) (start + batch.size()) / sequences.size() * 100;
                    double currentLoss = totalLoss / processedBatches;
                    double currentPerplexity = Math.exp(currentLoss);

                    System.out.printf("\r📊 Progreso: %.1f%% | Batches: %d/%d | Loss: %.4f | PPL: %.2f | Mem: %dGB",
                            progress, processedBatches, (sequences.size() + optimalBatchSize - 1) / optimalBatchSize,
                            currentLoss, currentPerplexity, systemInfo.getUsedMemoryGB());
                }
            }

            long epochTime = System.currentTimeMillis() - epochStart;
            double epochLoss = totalLoss / processedBatches;
            double epochPerplexity = Math.exp(epochLoss);

            System.out.printf("\n✅ Epoca completada en %.2fs | Loss: %.4f | Perplexity: %.2f\n",
                    epochTime / 1000.0, epochLoss, epochPerplexity);

            // Save checkpoint after each epoch without shutting down threadPool
            System.out.println("Creando checkpoint del modelo...");
            saveCheckpoint("checkpoint_epoch_" + (epoch + 1) + ".ser");

            if (epoch > 0 && epoch % 3 == 0) {
                learningRate *= 0.95;
                System.out.printf("📉 Learning rate ajustado a: %.6f\n", learningRate);
            }

            if (epoch % 5 == 0) {
                System.gc();
                Thread.yield();
            }
        }

        System.out.println("\n🎉 ¡Entrenamiento completado!");
        clean_old_checkpoints(epochs);
    }

    private void clean_old_checkpoints(int epochs) {
        String checkpointFileNames = "checkpoint_epoch_" + (epochs + 1) + ".ser";
        for (int i = 0; i < epochs; i++) {
            File ftmp = new File(checkpointFileNames);
            if (ftmp.exists()) {
                System.out.println("Checkpoint: "+i+" founded!\nCleaning!");
                ftmp.delete();
            }
            checkpointFileNames = "checkpoint_epoch_" + (i) + ".ser";
        }
    }

    private void saveCheckpoint(String filePath) {
        try (ObjectOutputStream out = new ObjectOutputStream(new FileOutputStream(filePath))) {
            out.writeObject(this);
            System.out.println("✅ Checkpoint guardado en: " + filePath);
        } catch (IOException e) {
            System.err.println("Error al guardar el checkpoint: " + e.getMessage());
        }
    }

    private double trainSequence(List<Integer> sequence) {
        int seqLen = sequence.size() - 1;
        List<LSTMState> states = new ArrayList<>();
        List<double[]> outputs = new ArrayList<>();
        List<double[]> inputs = new ArrayList<>();
        List<double[]> fs = new ArrayList<>();
        List<double[]> is = new ArrayList<>();
        List<double[]> os = new ArrayList<>();
        List<double[]> gs = new ArrayList<>();

        LSTMState state = new LSTMState(new double[hiddenSize], new double[hiddenSize]);
        double totalLoss = 0;

        for (int t = 0; t < seqLen; t++) {
            double[] x = getEmbedding(sequence.get(t));
            inputs.add(x);
            state = lstmStep(x, state, true);
            states.add(state);

            double[] f = sigmoid(addVectors(addVectors(matVecMul(Wf, x), matVecMul(Uf, state.h)), bf));
            double[] i = sigmoid(addVectors(addVectors(matVecMul(Wi, x), matVecMul(Ui, state.h)), bi));
            double[] o = sigmoid(addVectors(addVectors(matVecMul(Wo, x), matVecMul(Uo, state.h)), bo));
            double[] g = tanh(addVectors(addVectors(matVecMul(Wg, x), matVecMul(Ug, state.h)), bg));
            fs.add(f);
            is.add(i);
            os.add(o);
            gs.add(g);

            double[] output = softmax(addVectors(matVecMul(Why, state.h), by));
            outputs.add(output);

            int target = sequence.get(t + 1);
            totalLoss += -Math.log(Math.max(output[target], 1e-10));
        }

        backwardPass(sequence, states, outputs, inputs, fs, is, os, gs);

        return totalLoss / seqLen;
    }

    private void backwardPass(List<Integer> sequence, List<LSTMState> states, List<double[]> outputs, List<double[]> inputs, List<double[]> fs, List<double[]> is, List<double[]> os, List<double[]> gs) {
        double[][] dWf = new double[hiddenSize][embeddingSize];
        double[][] dWi = new double[hiddenSize][embeddingSize];
        double[][] dWo = new double[hiddenSize][embeddingSize];
        double[][] dWg = new double[hiddenSize][embeddingSize];
        double[][] dUf = new double[hiddenSize][hiddenSize];
        double[][] dUi = new double[hiddenSize][hiddenSize];
        double[][] dUo = new double[hiddenSize][hiddenSize];
        double[][] dUg = new double[hiddenSize][hiddenSize];
        double[] dbf = new double[hiddenSize];
        double[] dbi = new double[hiddenSize];
        double[] dbo = new double[hiddenSize];
        double[] dbg = new double[hiddenSize];
        double[][] dWe = new double[vocabSize][embeddingSize];
        double[][] dWhy = new double[vocabSize][hiddenSize];
        double[] dby = new double[vocabSize];

        double[] dhNext = new double[hiddenSize];
        double[] dcNext = new double[hiddenSize];

        for (int t = outputs.size() - 1; t >= 0; t--) {
            double[] output = outputs.get(t);
            int target = sequence.get(t + 1);
            LSTMState state = states.get(t);
            double[] x = inputs.get(t);
            double[] f = fs.get(t);
            double[] i = is.get(t);
            double[] o = os.get(t);
            double[] g = gs.get(t);
            double[] cPrev = (t > 0) ? states.get(t - 1).c : new double[hiddenSize];
            double[] hPrev = (t > 0) ? states.get(t - 1).h : new double[hiddenSize];

            double[] dOutput = output.clone();
            dOutput[target] -= 1.0;

            for (int k = 0; k < vocabSize; k++) {
                for (int j = 0; j < hiddenSize; j++) {
                    dWhy[k][j] += dOutput[k] * state.h[j];
                }
                dby[k] += dOutput[k];
            }

            double[] dh = new double[hiddenSize];
            for (int k = 0; k < vocabSize; k++) {
                for (int j = 0; j < hiddenSize; j++) {
                    dh[j] += dOutput[k] * Why[k][j];
                }
            }
            dh = addVectors(dh, dhNext);

            double[] dc = pointwiseMul(dh, o);
            dc = pointwiseMul(dc, dtanh(tanh(state.c)));
            dc = addVectors(dc, dcNext);

            double[] df = pointwiseMul(dc, cPrev);
            df = pointwiseMul(df, dsigmoid(f));
            double[] di = pointwiseMul(dc, g);
            di = pointwiseMul(di, dsigmoid(i));
            double[] dob = pointwiseMul(dh, tanh(state.c));
            dob = pointwiseMul(dob, dsigmoid(o));
            double[] dg = pointwiseMul(dc, i);
            dg = pointwiseMul(dg, dtanh(g));

            double[] dx = new double[embeddingSize];
            for (int j = 0; j < hiddenSize; j++) {
                for (int k = 0; k < embeddingSize; k++) {
                    dWf[j][k] += df[j] * x[k];
                    dWi[j][k] += di[j] * x[k];
                    dWo[j][k] += dob[j] * x[k];
                    dWg[j][k] += dg[j] * x[k];
                    dx[k] += df[j] * Wf[j][k] + di[j] * Wi[j][k] + dob[j] * Wo[j][k] + dg[j] * Wg[j][k];
                }
                dbf[j] += df[j];
                dbi[j] += di[j];
                dbo[j] += dob[j];
                dbg[j] += dg[j];
            }

            for (int j = 0; j < hiddenSize; j++) {
                for (int k = 0; k < hiddenSize; k++) {
                    dUf[j][k] += df[j] * hPrev[k];
                    dUi[j][k] += di[j] * hPrev[k];
                    dUo[j][k] += dob[j] * hPrev[k];
                    dUg[j][k] += dg[j] * hPrev[k];
                }
            }

            if (t < outputs.size() - 1) {
                dWe[sequence.get(t)] = addVectors(dWe[sequence.get(t)], dx);
            }

            dhNext = new double[hiddenSize];
            for (int j = 0; j < hiddenSize; j++) {
                for (int k = 0; k < hiddenSize; k++) {
                    dhNext[k] += df[j] * Uf[j][k] + di[j] * Ui[j][k] + dob[j] * Uo[j][k] + dg[j] * Ug[j][k];
                }
            }
            dcNext = pointwiseMul(dc, f);
        }

        double norm = computeGradientNorm(dbf, dbi, dbo, dbg, dby, dWf, dWi, dWo, dWg, dUf, dUi, dUo, dUg, dWe, dWhy);
        if (norm > gradientClipNorm) {
            double scale = gradientClipNorm / norm;
            scaleMatrix(dWf, scale);
            scaleMatrix(dWi, scale);
            scaleMatrix(dWo, scale);
            scaleMatrix(dWg, scale);
            scaleMatrix(dUf, scale);
            scaleMatrix(dUi, scale);
            scaleMatrix(dUo, scale);
            scaleMatrix(dUg, scale);
            scaleVector(dbf, scale);
            scaleVector(dbi, scale);
            scaleVector(dbo, scale);
            scaleVector(dbg, scale);
            scaleMatrix(dWe, scale);
            scaleMatrix(dWhy, scale);
            scaleVector(dby, scale);
        }

        updateWeights(dWf, dWi, dWo, dWg, dUf, dUi, dUo, dUg, dbf, dbi, dbo, dbg, dWe, dWhy, dby);
    }

    private void updateWeights(double[][] dWf, double[][] dWi, double[][] dWo, double[][] dWg,
            double[][] dUf, double[][] dUi, double[][] dUo, double[][] dUg,
            double[] dbf, double[] dbi, double[] dbo, double[] dbg,
            double[][] dWe, double[][] dWhy, double[] dby) {
        for (int i = 0; i < hiddenSize; i++) {
            for (int j = 0; j < embeddingSize; j++) {
                Wf[i][j] -= learningRate * dWf[i][j];
                Wi[i][j] -= learningRate * dWi[i][j];
                Wo[i][j] -= learningRate * dWo[i][j];
                Wg[i][j] -= learningRate * dWg[i][j];
            }
            for (int j = 0; j < hiddenSize; j++) {
                Uf[i][j] -= learningRate * dUf[i][j];
                Ui[i][j] -= learningRate * dUi[i][j];
                Uo[i][j] -= learningRate * dUo[i][j];
                Ug[i][j] -= learningRate * dUg[i][j];
            }
            bf[i] -= learningRate * dbf[i];
            bi[i] -= learningRate * dbi[i];
            bo[i] -= learningRate * dbo[i];
            bg[i] -= learningRate * dbg[i];
        }
        for (int i = 0; i < vocabSize; i++) {
            for (int j = 0; j < embeddingSize; j++) {
                We[i][j] -= learningRate * dWe[i][j];
            }
            for (int j = 0; j < hiddenSize; j++) {
                Why[i][j] -= learningRate * dWhy[i][j];
            }
            by[i] -= learningRate * dby[i];
        }
    }

    private double computeGradientNorm(double[] dbf, double[] dbi, double[] dbo, double[] dbg, double[] dby,
            double[][] dWf, double[][] dWi, double[][] dWo, double[][] dWg,
            double[][] dUf, double[][] dUi, double[][] dUo, double[][] dUg,
            double[][] dWe, double[][] dWhy) {
        double norm = 0;

        // Sum norm for 1D arrays
        for (double val : dbf) {
            norm += val * val;
        }
        for (double val : dbi) {
            norm += val * val;
        }
        for (double val : dbo) {
            norm += val * val;
        }
        for (double val : dbg) {
            norm += val * val;
        }
        for (double val : dby) {
            norm += val * val;
        }

        // Sum norm for 2D arrays
        for (double[][] matrix : new double[][][]{dWf, dWi, dWo, dWg, dUf, dUi, dUo, dUg, dWe, dWhy}) {
            for (double[] row : matrix) {
                for (double val : row) {
                    norm += val * val;
                }
            }
        }

        return Math.sqrt(norm);
    }

    private void scaleMatrix(double[][] matrix, double scale) {
        for (int i = 0; i < matrix.length; i++) {
            for (int j = 0; j < matrix[i].length; j++) {
                matrix[i][j] *= scale;
            }
        }
    }

    private void scaleVector(double[] x, double scale) {
        for (int i = 0; i < x.length; i++) {
            x[i] *= scale;
        }
    }

    public String generateText(String seed, int maxLength, double temperature) {
        StringBuilder result = new StringBuilder();
        LSTMState state = new LSTMState(new double[hiddenSize], new double[hiddenSize]);

        String currentWord = seed.toLowerCase();
        if (!wordToIndex.containsKey(currentWord)) {
            currentWord = "<START>";
        }

        result.append(currentWord);

        for (int i = 0; i < maxLength; i++) {
            double[] x = getEmbedding(wordToIndex.get(currentWord));
            state = lstmStep(x, state, false);

            double[] logits = addVectors(matVecMul(Why, state.h), by);

            for (int j = 0; j < logits.length; j++) {
                logits[j] /= temperature;
            }

            double[] probs = softmax(logits);
            int nextIdx = sampleFromDistribution(probs);

            currentWord = indexToWord.get(nextIdx);

            if ("<END>".equals(currentWord)) {
                break;
            }

            result.append(" ").append(currentWord);
        }

        return result.toString();
    }

    private void optimizeForHardware() {
        long availableMemoryMB = systemInfo.getAvailableMemoryMB();
        int cores = systemInfo.getAvailableCores();

        System.out.println("🔧 OPTIMIZACION AUTOMATICA DE HARDWARE:");
        systemInfo.printSystemInfo();

        optimalThreads = Math.min(cores - 2, 16);
        optimalThreads = Math.max(optimalThreads, 2);

        int maxBatchByMemory = (int) Math.min(availableMemoryMB / 2, 512);
        optimalBatchSize = Math.min(maxBatchByMemory, 128);
        optimalBatchSize = Math.max(optimalBatchSize, 16);

        if (threadPool == null || threadPool.isShutdown()) {
            threadPool = Executors.newFixedThreadPool(optimalThreads);
        }

        System.out.printf("✅ Configuracion optimizada - Threads: %d, Batch Size: %d\n\n",
                optimalThreads, optimalBatchSize);
    }

    private List<List<Integer>> prepareTrainingData(int seqLength) {
        List<List<Integer>> sequences = new ArrayList<>();

        for (int start = 0; start < corpusWords.size() - seqLength - 1; start += seqLength / 2) {
            if (start + seqLength + 1 >= corpusWords.size()) {
                break;
            }

            List<Integer> sequence = new ArrayList<>();
            boolean validSeq = true;

            for (int i = 0; i <= seqLength; i++) {
                String word = corpusWords.get(start + i);
                if (!wordToIndex.containsKey(word)) {
                    sequence.add(wordToIndex.get("<UNK>"));
                } else {
                    sequence.add(wordToIndex.get(word));
                }
            }

            if (validSeq) {
                sequences.add(sequence);
            }
        }

        return sequences;
    }

    private double processBatchParallel(List<List<Integer>> batch) {
        // Ensure threadPool is initialized
        if (threadPool == null || threadPool.isShutdown()) {
            threadPool = Executors.newFixedThreadPool(optimalThreads);
        }

        if (batch.size() == 1) {
            return trainSequence(batch.get(0));
        }

        int chunkSize = Math.max(1, batch.size() / optimalThreads);
        List<Future<Double>> futures = new ArrayList<>();

        for (int i = 0; i < batch.size(); i += chunkSize) {
            int end = Math.min(i + chunkSize, batch.size());
            List<List<Integer>> chunk = batch.subList(i, end);

            futures.add(threadPool.submit(() -> {
                double chunkLoss = 0;
                for (List<Integer> sequence : chunk) {
                    chunkLoss += trainSequence(sequence);
                }
                return chunkLoss / chunk.size();
            }));
        }

        double totalLoss = 0;
        int completedTasks = 0;

        for (Future<Double> future : futures) {
            try {
                totalLoss += future.get();
                completedTasks++;
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                System.err.println("Error en procesamiento paralelo: " + e.getMessage());
            } catch (ExecutionException e) {
                System.err.println("Error en procesamiento paralelo: " + e.getMessage());
            }
        }

        return completedTasks > 0 ? totalLoss / completedTasks : 0;
    }

    private double[] getEmbedding(int wordIdx) {
        if (wordIdx >= 0 && wordIdx < vocabSize && We[wordIdx] != null) {
            return We[wordIdx].clone();
        }
        Integer unkIdx = wordToIndex.get("<UNK>");
        if (unkIdx == null || unkIdx >= vocabSize || We[unkIdx] == null) {
            throw new IllegalStateException("Token <UNK> no está en el vocabulario o no está inicializado");
        }
        return We[unkIdx].clone();
    }

    private double[] sigmoid(double[] x) {
        double[] result = new double[x.length];
        for (int i = 0; i < x.length; i++) {
            result[i] = 1.0 / (1.0 + Math.exp(-x[i]));
        }
        return result;
    }

    private double[] dsigmoid(double[] x) {
        double[] result = new double[x.length];
        for (int i = 0; i < x.length; i++) {
            double sig = 1.0 / (1.0 + Math.exp(-x[i]));
            result[i] = sig * (1.0 - sig);
        }
        return result;
    }

    private double[] tanh(double[] x) {
        double[] result = new double[x.length];
        for (int i = 0; i < x.length; i++) {
            result[i] = Math.tanh(x[i]);
        }
        return result;
    }

    private double[] dtanh(double[] x) {
        double[] result = new double[x.length];
        for (int i = 0; i < x.length; i++) {
            double tanh = Math.tanh(x[i]);
            result[i] = 1.0 - tanh * tanh;
        }
        return result;
    }

    private double[] softmax(double[] x) {
        double max = Arrays.stream(x).max().orElse(0.0);
        double sum = 0;
        double[] result = new double[x.length];

        for (int i = 0; i < x.length; i++) {
            result[i] = Math.exp(x[i] - max);
            sum += result[i];
        }

        for (int i = 0; i < x.length; i++) {
            result[i] /= sum;
        }

        return result;
    }

    private double[] addVectors(double[] a, double[] b) {
        double[] result = new double[a.length];
        for (int i = 0; i < a.length; i++) {
            result[i] = a[i] + b[i];
        }
        return result;
    }

    private double[] pointwiseMul(double[] a, double[] b) {
        double[] result = new double[a.length];
        for (int i = 0; i < a.length; i++) {
            result[i] = a[i] * b[i];
        }
        return result;
    }

    private double[] matVecMul(double[][] matrix, double[] vector) {
        double[] result = new double[matrix.length];
        for (int i = 0; i < matrix.length; i++) {
            for (int j = 0; j < vector.length; j++) {
                result[i] += matrix[i][j] * vector[j];
            }
        }
        return result;
    }

    private double[] applyDropout(double[] x, double rate) {
        double[] result = new double[x.length];
        for (int i = 0; i < x.length; i++) {
            if (rand.nextDouble() > rate) {
                result[i] = x[i] / (1.0 - rate);
            }
        }
        return result;
    }

    private double vectorNorm(double[] x) {
        double sum = 0;
        for (double v : x) {
            sum += v * v;
        }
        return Math.sqrt(sum);
    }

    private int sampleFromDistribution(double[] probs) {
        double r = rand.nextDouble();
        double sum = 0;
        for (int i = 0; i < probs.length; i++) {
            sum += probs[i];
            if (r <= sum) {
                return i;
            }
        }
        return probs.length - 1;
    }

    private void cleanup() {
        if (threadPool != null && !threadPool.isShutdown()) {
            threadPool.shutdown();
            try {
                if (!threadPool.awaitTermination(5, TimeUnit.SECONDS)) {
                    threadPool.shutdownNow();
                }
            } catch (InterruptedException e) {
                threadPool.shutdownNow();
                Thread.currentThread().interrupt();
            }
        }
    }

    private void readObject(ObjectInputStream in) throws IOException, ClassNotFoundException {
        in.defaultReadObject();
        threadPool = Executors.newFixedThreadPool(optimalThreads);
    }

    public void saveModel(String filePath) {
        cleanup();
        try (ObjectOutputStream out = new ObjectOutputStream(new FileOutputStream(filePath))) {
            out.writeObject(this);
            System.out.println("Modelo guardado en: " + filePath);
        } catch (IOException e) {
            System.err.println("Error al guardar el modelo: " + e.getMessage());
        }
    }

    public static ImprovedLanguageModel loadModel(String filePath) {
        try (ObjectInputStream in = new ObjectInputStream(new FileInputStream(filePath))) {
            ImprovedLanguageModel model = (ImprovedLanguageModel) in.readObject();
            return model;
        } catch (IOException | ClassNotFoundException e) {
            System.err.println("Error al cargar el modelo: " + e.getMessage());
            return null;
        }
    }

    public void startInteractiveMode() throws IOException {
        Scanner scanner = new Scanner(System.in);
        System.out.println("=== MODELO DE LENGUAJE MEJORADO CON OPTIMIZACION DE HARDWARE ===");
        System.out.println("Comandos disponibles:");
        System.out.println("  EXIT - Salir");
        System.out.println("  LOAD_TEXT - Cargar corpus de texto");
        System.out.println("  TRAIN - Entrenar modelo");
        System.out.println("  SAVE - Guardar modelo");
        System.out.println("  LOAD - Cargar modelo");
        System.out.println("  STATUS - Ver estado del sistema");
        System.out.println("  OPTIMIZE - Reoptimizar para hardware");

        while (true) {
            System.out.print("\n> ");
            String input = scanner.nextLine().trim();

            if (input.equalsIgnoreCase("EXIT")) {
                cleanup();
                break;
            } else if (input.equalsIgnoreCase("STATUS")) {
                systemInfo.printSystemInfo();
                System.out.printf("Configuracion actual - Threads: %d, Batch Size: %d\n",
                        optimalThreads, optimalBatchSize);
                if (vocabSize > 0) {
                    System.out.printf("Modelo - Vocabulario: %d palabras, Corpus: %d palabras\n",
                            vocabSize, corpusWords.size());
                }
            } else if (input.equalsIgnoreCase("OPTIMIZE")) {
                optimizeForHardware();
            } else if (input.equalsIgnoreCase("LOAD_TEXT")) {
                System.out.print("📁 Archivo: (por defecto: src/javallmsimple/corpus.txt)");
                String path = scanner.nextLine().trim();
                if (path.isEmpty()) {
                    path = "src/javallmsimple/corpus.txt";
                }
                try {
                    System.out.println("🔄 Cargando corpus...");
                    long startTime = System.currentTimeMillis();
                    loadCorpus(path);
                    long loadTime = System.currentTimeMillis() - startTime;
                    System.out.printf("✅ Corpus cargado en %.2fs\n", loadTime / 1000.0);
                    System.out.printf("📊 Estadisticas: %d palabras, vocabulario de %d terminos\n",
                            corpusWords.size(), vocabSize);
                } catch (IOException e) {
                    System.out.println("❌ Error: " + e.getMessage());
                }
            } else if (input.equalsIgnoreCase("TRAIN")) {
                if (vocabSize == 0) {
                    System.out.println("⚠️  Primero carga un corpus con LOAD_TEXT");
                    continue;
                }
                System.out.print("🎯 Epocas (recomendado 15-25): ");
                String epochStr = scanner.nextLine().trim();
                int epochs = epochStr.isEmpty() ? 20 : Integer.parseInt(epochStr);

                System.out.print("📏 Longitud de secuencia (recomendado 12-20): ");
                String seqStr = scanner.nextLine().trim();
                int seqLength = seqStr.isEmpty() ? 15 : Integer.parseInt(seqStr);

                long trainStart = System.currentTimeMillis();
                train(epochs, seqLength);
                long trainTime = System.currentTimeMillis() - trainStart;

                System.out.printf("\n🏁 Entrenamiento completado en %.2f minutos\n",
                        trainTime / 60000.0);
            } else if (input.equalsIgnoreCase("SAVE")) {
                System.out.print("💾 Nombre del archivo (default: train_model.ser): ");
                String filename = scanner.nextLine().trim();
                if (filename.isEmpty()) {
                    filename = "train_model.ser";
                }
                saveModel(filename);
            } else if (input.equalsIgnoreCase("LOAD")) {
                System.out.print("📂 Archivo a cargar: ");
                String filename = scanner.nextLine().trim();
                ImprovedLanguageModel loaded = loadModel(filename);
                if (loaded != null) {
                    System.out.println("✅ Modelo cargado exitosamente");
                    loaded.startInteractiveMode();
                    return;
                }
            } else if (!input.isEmpty()) {
                if (vocabSize == 0) {
                    System.out.println("⚠️  Modelo no entrenado. Usa LOAD_TEXT y TRAIN primero.");
                    continue;
                }

                System.out.print("📝 Longitud del texto (default 30): ");
                String lenStr = scanner.nextLine().trim();
                int length = lenStr.isEmpty() ? 30 : Integer.parseInt(lenStr);

                System.out.print("🌡️  Temperatura (0.1=conservador, 1.5=creativo, default 0.8): ");
                String tempStr = scanner.nextLine().trim();
                double temp = tempStr.isEmpty() ? 0.8 : Double.parseDouble(tempStr);

                System.out.println("🤖 Generando texto...");
                long genStart = System.currentTimeMillis();
                String response = generateText(input, length, temp);
                long genTime = System.currentTimeMillis() - genStart;

                System.out.printf("\n📜 Generado en %dms:\n", genTime);
                System.out.println("\"" + response + "\"");
            }
        }
        scanner.close();
    }

    public static void main(String[] args) {
        System.out.println("🚀 Iniciando Modelo de Lenguaje Optimizado...\n");

        ImprovedLanguageModel model = new ImprovedLanguageModel();

        Runtime.getRuntime().addShutdownHook(new Thread(() -> {
            System.out.println("\n🔄 Cerrando modelo y liberando recursos...");
            model.cleanup();
        }));

        try {
            model.startInteractiveMode();
        } catch (IOException e) {
            System.err.println("❌ Error: " + e.getMessage());
        } finally {
            model.cleanup();
        }
    }

    private static class SystemInfo implements Serializable {

        private static final long serialVersionUID = 1L;

        public SystemInfo() {
        }

        private void printSystemInfo() {
            Runtime runtime = Runtime.getRuntime();
            long maxMemory = runtime.maxMemory() / (1024 * 1024);
            long usedMemory = (runtime.totalMemory() - runtime.freeMemory()) / (1024 * 1024);
            Properties p = System.getProperties();
            System.out.println("OS: " + p.getProperty("os.name"));
            System.out.println("OS Arch: " + p.getProperty("os.arch"));
            System.out.println("RAM Total: " + maxMemory + "MB");
            System.out.println("RAM Used: " + usedMemory + "MB");
            System.out.println("Available Cores: " + runtime.availableProcessors());
        }

        public int getUsedMemoryGB() {
            Runtime runtime = Runtime.getRuntime();
            long usedMemoryBytes = runtime.totalMemory() - runtime.freeMemory();
            return (int) (usedMemoryBytes / (1024L * 1024L * 1024L));
        }

        private long getAvailableMemoryMB() {
            Runtime runtime = Runtime.getRuntime();
            return runtime.maxMemory() / (1024 * 1024);
        }

        private int getAvailableCores() {
            return Runtime.getRuntime().availableProcessors();
        }
    }
}
