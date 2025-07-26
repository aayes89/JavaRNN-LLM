/*
 * The MIT License
 *
 * Copyright 2025 Slam.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */
package javallm;

import java.io.*;
import java.util.*;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.stream.Collectors;

public class SpanishPowerPCLLM implements Serializable {
    private static final long serialVersionUID = 1L;
    private Map<String, Integer> wordToIndex;
    private Map<Integer, String> indexToWord;
    private int vocabSize;
    private double[][] We;    // Embeddings de palabras
    private double[][] Wxh1;  // Pesos embedding -> oculta1
    private double[][] Whh1;  // Pesos oculta1 -> oculta1
    private double[][] Wxh2;  // Pesos oculta1 -> oculta2
    private double[][] Whh2;  // Pesos oculta2 -> oculta2
    private double[][] Why;   // Pesos oculta2 -> salida
    private double[] bh1;     // Bias oculta1
    private double[] bh2;     // Bias oculta2
    private double[] by;      // Bias salida
    private int hiddenSize = 512; // Aumentado para mayor capacidad
    private int embeddingSize = 128; // Tamaño de embeddings
    private transient Random rand;
    private List<String> conversation;
    private double learningRate = 0.001;
    private double[][] adamMWe, adamVWe, adamM1, adamV1, adamM2, adamV2, adamMy, adamVy;
    private double beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8;
    private int t = 0;
    private static final int MAX_VOCAB_SIZE = 10000;
    private double dropoutRate = 0.2; // Regularización

    public SpanishPowerPCLLM() {
        wordToIndex = new HashMap<>();
        indexToWord = new HashMap<>();
        vocabSize = 0;
        rand = new Random();
        conversation = new ArrayList<>();
    }

    private void readObject(ObjectInputStream in) throws IOException, ClassNotFoundException {
        in.defaultReadObject();
        rand = new Random();
    }

    // Cargar corpus
    public void loadCorpus(String... filePaths) throws IOException {
        List<String> words = new ArrayList<>();
        Map<String, Integer> wordCounts = new HashMap<>();

        for (String filePath : filePaths) {
            try (BufferedReader br = new BufferedReader(new FileReader(filePath))) {
                String line;
                while ((line = br.readLine()) != null) {
                    String[] tokens = line.toLowerCase()
                            .replaceAll("[^a-záéíóúñü ]", "")
                            .split("\\s+");
                    for (String token : tokens) {
                        if (!token.isEmpty()) {
                            words.add(token);
                            wordCounts.put(token, wordCounts.getOrDefault(token, 0) + 1);
                        }
                    }
                }
            }
        }

        List<Map.Entry<String, Integer>> sortedWords = wordCounts.entrySet().stream()
                .sorted((e1, e2) -> e2.getValue().compareTo(e1.getValue()))
                .limit(MAX_VOCAB_SIZE)
                .collect(Collectors.toList());

        wordToIndex.clear();
        indexToWord.clear();
        vocabSize = 0;
        for (Map.Entry<String, Integer> entry : sortedWords) {
            wordToIndex.put(entry.getKey(), vocabSize);
            indexToWord.put(vocabSize, entry.getKey());
            vocabSize++;
        }

        if (Wxh1 == null) {
            We = new double[vocabSize][embeddingSize];
            Wxh1 = new double[hiddenSize][embeddingSize];
            Whh1 = new double[hiddenSize][hiddenSize];
            Wxh2 = new double[hiddenSize][hiddenSize];
            Whh2 = new double[hiddenSize][hiddenSize];
            Why = new double[vocabSize][hiddenSize];
            bh1 = new double[hiddenSize];
            bh2 = new double[hiddenSize];
            by = new double[vocabSize];

            adamMWe = new double[vocabSize][embeddingSize];
            adamVWe = new double[vocabSize][embeddingSize];
            adamM1 = new double[hiddenSize][embeddingSize];
            adamV1 = new double[hiddenSize][embeddingSize];
            adamM2 = new double[hiddenSize][hiddenSize];
            adamV2 = new double[hiddenSize][hiddenSize];
            adamMy = new double[vocabSize][hiddenSize];
            adamVy = new double[vocabSize][hiddenSize];

            for (int i = 0; i < vocabSize; i++) {
                for (int j = 0; j < embeddingSize; j++) {
                    We[i][j] = rand.nextGaussian() * 0.01;
                }
            }
            for (int i = 0; i < hiddenSize; i++) {
                for (int j = 0; j < embeddingSize; j++) {
                    Wxh1[i][j] = rand.nextGaussian() * 0.01;
                }
                for (int j = 0; j < hiddenSize; j++) {
                    Whh1[i][j] = rand.nextGaussian() * 0.01;
                    Wxh2[i][j] = rand.nextGaussian() * 0.01;
                    Whh2[i][j] = rand.nextGaussian() * 0.01;
                }
                bh1[i] = 0;
                bh2[i] = 0;
            }
            for (int i = 0; i < vocabSize; i++) {
                for (int j = 0; j < hiddenSize; j++) {
                    Why[i][j] = rand.nextGaussian() * 0.01;
                }
                by[i] = 0;
            }
        }
    }

    // Forward pass con atención simplificada
    private double[][][] forward(double[][] xHistory, double[] h1Prev, double[] h2Prev, int seqLength) {
        double[][] h1History = new double[seqLength][hiddenSize];
        double[][] h2History = new double[seqLength][hiddenSize];
        double[][] yHistory = new double[seqLength][vocabSize];
        double[][] attentionWeights = new double[seqLength][seqLength];

        double[] h1 = h1Prev.clone();
        double[] h2 = h2Prev.clone();

        for (int t = 0; t < seqLength; t++) {
            // Embedding
            double[] x = xHistory[t];
            double[] emb = new double[embeddingSize];
            for (int i = 0; i < vocabSize; i++) {
                if (x[i] > 0) {
                    for (int j = 0; j < embeddingSize; j++) {
                        emb[j] = We[i][j];
                    }
                    break;
                }
            }

            // Capa 1
            double[] h1New = new double[hiddenSize];
            for (int i = 0; i < hiddenSize; i++) {
                double sum = bh1[i];
                for (int j = 0; j < embeddingSize; j++) {
                    sum += Wxh1[i][j] * emb[j];
                }
                for (int j = 0; j < hiddenSize; j++) {
                    sum += Whh1[i][j] * h1[j];
                }
                h1New[i] = Math.tanh(sum) * (rand.nextDouble() > dropoutRate ? 1 : 0);
            }
            h1History[t] = h1New;

            // Capa 2
            double[] h2New = new double[hiddenSize];
            for (int i = 0; i < hiddenSize; i++) {
                double sum = bh2[i];
                for (int j = 0; j < hiddenSize; j++) {
                    sum += Wxh2[i][j] * h1New[j];
                    sum += Whh2[i][j] * h2[j];
                }
                h2New[i] = Math.tanh(sum) * (rand.nextDouble() > dropoutRate ? 1 : 0);
            }
            h2History[t] = h2New;

            // Atención simplificada
            double[] attentionContext = new double[hiddenSize];
            double[] attScores = new double[seqLength];
            double attSum = 0;
            for (int s = 0; s <= t; s++) {
                double score = 0;
                for (int i = 0; i < hiddenSize; i++) {
                    score += h2New[i] * (s > 0 ? h2History[s-1][i] : h2[i]);
                }
                attScores[s] = Math.exp(score);
                attSum += attScores[s];
            }
            for (int s = 0; s <= t; s++) {
                attentionWeights[t][s] = attSum > 0 ? attScores[s] / attSum : 0;
                for (int i = 0; i < hiddenSize; i++) {
                    attentionContext[i] += attentionWeights[t][s] * (s > 0 ? h2History[s-1][i] : h2[i]);
                }
            }

            // Salida
            double[] y = new double[vocabSize];
            double maxY = Double.NEGATIVE_INFINITY;
            for (int i = 0; i < vocabSize; i++) {
                double sum = by[i];
                for (int j = 0; j < hiddenSize; j++) {
                    sum += Why[i][j] * attentionContext[j];
                }
                y[i] = sum;
                if (sum > maxY) maxY = sum;
            }
            double sumExp = 0;
            for (int i = 0; i < vocabSize; i++) {
                y[i] = Math.exp(y[i] - maxY);
                sumExp += y[i];
            }
            for (int i = 0; i < vocabSize; i++) {
                y[i] /= sumExp;
            }
            yHistory[t] = y;

            h1 = h1New;
            h2 = h2New;
        }

        return new double[][][]{h1History, h2History, yHistory, attentionWeights};
    }

    // Backward pass
    private void backward(double[][] xHistory, double[][] yTrueHistory, double[][][] forwardResult, int seqLength) {
        double[][] h1History = forwardResult[0];
        double[][] h2History = forwardResult[1];
        double[][] yHistory = forwardResult[2];
        double[][] attentionWeights = forwardResult[3];

        double[][] dWe = new double[vocabSize][embeddingSize];
        double[][] dWxh1 = new double[hiddenSize][embeddingSize];
        double[][] dWhh1 = new double[hiddenSize][hiddenSize];
        double[][] dWxh2 = new double[hiddenSize][hiddenSize];
        double[][] dWhh2 = new double[hiddenSize][hiddenSize];
        double[][] dWhy = new double[vocabSize][hiddenSize];
        double[] dbh1 = new double[hiddenSize];
        double[] dbh2 = new double[hiddenSize];
        double[] dby = new double[vocabSize];
        double[] dh1Next = new double[hiddenSize];
        double[] dh2Next = new double[hiddenSize];

        for (int t = seqLength - 1; t >= 0; t--) {
            double[] yPred = yHistory[t];
            double[] yTrue = yTrueHistory[t];
            double[] h2 = h2History[t];
            double[] h1 = h1History[t];
            double[] x = xHistory[t];
            double[] emb = new double[embeddingSize];
            int wordIdx = -1;
            for (int i = 0; i < vocabSize; i++) {
                if (x[i] > 0) {
                    wordIdx = i;
                    for (int j = 0; j < embeddingSize; j++) {
                        emb[j] = We[i][j];
                    }
                    break;
                }
            }

            double[] dy = new double[vocabSize];
            for (int i = 0; i < vocabSize; i++) {
                dy[i] = yPred[i] - yTrue[i];
            }

            double[] dAttentionContext = new double[hiddenSize];
            for (int i = 0; i < vocabSize; i++) {
                for (int j = 0; j < hiddenSize; j++) {
                    dWhy[i][j] += dy[i] * attentionWeights[t][t] * h2[j];
                    dAttentionContext[j] += dy[i] * Why[i][j];
                }
                dby[i] += dy[i];
            }

            double[] dh2 = new double[hiddenSize];
            for (int i = 0; i < hiddenSize; i++) {
                dh2[i] = dAttentionContext[i] * (1 - h2[i] * h2[i]);
                dh2[i] += dh2Next[i];
            }

            for (int i = 0; i < hiddenSize; i++) {
                for (int j = 0; j < hiddenSize; j++) {
                    dWxh2[i][j] += dh2[i] * h1[j];
                    dWhh2[i][j] += dh2[i] * (t > 0 ? h2History[t-1][j] : 0);
                }
                dbh2[i] += dh2[i];
            }

            double[] dh1 = new double[hiddenSize];
            for (int i = 0; i < hiddenSize; i++) {
                for (int j = 0; j < hiddenSize; j++) {
                    dh1[i] += Wxh2[j][i] * dh2[j];
                }
                dh1[i] *= (1 - h1[i] * h1[i]);
                dh1[i] += dh1Next[i];
            }

            for (int i = 0; i < hiddenSize; i++) {
                for (int j = 0; j < embeddingSize; j++) {
                    dWxh1[i][j] += dh1[i] * emb[j];
                }
                for (int j = 0; j < hiddenSize; j++) {
                    dWhh1[i][j] += dh1[i] * (t > 0 ? h1History[t-1][j] : 0);
                }
                dbh1[i] += dh1[i];
            }

            if (wordIdx >= 0) {
                for (int i = 0; i < hiddenSize; i++) {
                    for (int j = 0; j < embeddingSize; j++) {
                        dWe[wordIdx][j] += dh1[i] * Wxh1[i][j];
                    }
                }
            }

            dh1Next = dh1;
            dh2Next = dh2;
        }

        t++;
        updateAdam(We, dWe, adamMWe, adamVWe);
        updateAdam(Wxh1, dWxh1, adamM1, adamV1);
        updateAdam(Whh1, dWhh1, adamM2, adamV2);
        updateAdam(Wxh2, dWxh2, new double[hiddenSize][hiddenSize], new double[hiddenSize][hiddenSize]);
        updateAdam(Whh2, dWhh2, new double[hiddenSize][hiddenSize], new double[hiddenSize][hiddenSize]);
        updateAdam(Why, dWhy, adamMy, adamVy);
        for (int i = 0; i < hiddenSize; i++) {
            bh1[i] -= learningRate * dbh1[i];
            bh2[i] -= learningRate * dbh2[i];
        }
        for (int i = 0; i < vocabSize; i++) {
            by[i] -= learningRate * dby[i];
        }
    }

    // Backward paralelo
    private Object[] backwardParallel(double[][] xHistory, double[][] yTrueHistory, double[][][] forwardResult, int seqLength) {
        double[][] h1History = forwardResult[0];
        double[][] h2History = forwardResult[1];
        double[][] yHistory = forwardResult[2];
        double[][] attentionWeights = forwardResult[3];

        double[][] dWe = new double[vocabSize][embeddingSize];
        double[][] dWxh1 = new double[hiddenSize][embeddingSize];
        double[][] dWhh1 = new double[hiddenSize][hiddenSize];
        double[][] dWxh2 = new double[hiddenSize][hiddenSize];
        double[][] dWhh2 = new double[hiddenSize][hiddenSize];
        double[][] dWhy = new double[vocabSize][hiddenSize];
        double[] dbh1 = new double[hiddenSize];
        double[] dbh2 = new double[hiddenSize];
        double[] dby = new double[vocabSize];
        double[] dh1Next = new double[hiddenSize];
        double[] dh2Next = new double[hiddenSize];

        for (int t = seqLength - 1; t >= 0; t--) {
            double[] yPred = yHistory[t];
            double[] yTrue = yTrueHistory[t];
            double[] h2 = h2History[t];
            double[] h1 = h1History[t];
            double[] x = xHistory[t];
            double[] emb = new double[embeddingSize];
            int wordIdx = -1;
            for (int i = 0; i < vocabSize; i++) {
                if (x[i] > 0) {
                    wordIdx = i;
                    for (int j = 0; j < embeddingSize; j++) {
                        emb[j] = We[i][j];
                    }
                    break;
                }
            }

            double[] dy = new double[vocabSize];
            for (int i = 0; i < vocabSize; i++) {
                dy[i] = yPred[i] - yTrue[i];
            }

            double[] dAttentionContext = new double[hiddenSize];
            for (int i = 0; i < vocabSize; i++) {
                for (int j = 0; j < hiddenSize; j++) {
                    dWhy[i][j] += dy[i] * attentionWeights[t][t] * h2[j];
                    dAttentionContext[j] += dy[i] * Why[i][j];
                }
                dby[i] += dy[i];
            }

            double[] dh2 = new double[hiddenSize];
            for (int i = 0; i < hiddenSize; i++) {
                dh2[i] = dAttentionContext[i] * (1 - h2[i] * h2[i]);
                dh2[i] += dh2Next[i];
            }

            for (int i = 0; i < hiddenSize; i++) {
                for (int j = 0; j < hiddenSize; j++) {
                    dWxh2[i][j] += dh2[i] * h1[j];
                    dWhh2[i][j] += dh2[i] * (t > 0 ? h2History[t-1][j] : 0);
                }
                dbh2[i] += dh2[i];
            }

            double[] dh1 = new double[hiddenSize];
            for (int i = 0; i < hiddenSize; i++) {
                for (int j = 0; j < hiddenSize; j++) {
                    dh1[i] += Wxh2[j][i] * dh2[j];
                }
                dh1[i] *= (1 - h1[i] * h1[i]);
                dh1[i] += dh1Next[i];
            }

            for (int i = 0; i < hiddenSize; i++) {
                for (int j = 0; j < embeddingSize; j++) {
                    dWxh1[i][j] += dh1[i] * emb[j];
                }
                for (int j = 0; j < hiddenSize; j++) {
                    dWhh1[i][j] += dh1[i] * (t > 0 ? h1History[t-1][j] : 0);
                }
                dbh1[i] += dh1[i];
            }

            if (wordIdx >= 0) {
                for (int i = 0; i < hiddenSize; i++) {
                    for (int j = 0; j < embeddingSize; j++) {
                        dWe[wordIdx][j] += dh1[i] * Wxh1[i][j];
                    }
                }
            }

            dh1Next = dh1;
            dh2Next = dh2;
        }

        return new Object[]{dWe, dWxh1, dWhh1, dWxh2, dWhh2, dWhy, dbh1, dbh2, dby};
    }

    private void updateAdam(double[][] W, double[][] dW, double[][] m, double[][] v) {
        for (int i = 0; i < W.length; i++) {
            for (int j = 0; j < W[0].length; j++) {
                m[i][j] = beta1 * m[i][j] + (1 - beta1) * dW[i][j];
                v[i][j] = beta2 * v[i][j] + (1 - beta2) * dW[i][j] * dW[i][j];
                double mHat = m[i][j] / (1 - Math.pow(beta1, t));
                double vHat = v[i][j] / (1 - Math.pow(beta2, t));
                W[i][j] -= learningRate * mHat / (Math.sqrt(vHat) + epsilon);
            }
        }
    }

    // Entrenamiento paralelo
    public void train(List<String> words, int epochs, int batchSize, int seqLength) throws IOException, InterruptedException {
        System.out.println("Vocab size: " + vocabSize);
        double prevLoss = Double.POSITIVE_INFINITY;
        ExecutorService executor = Executors.newFixedThreadPool(24); // 24 hilos

        // Clase interna para acumular resultados por hilo
        class BatchResult {
            double loss;
            int validPairs;
            double[][] dWe;
            double[][] dWxh1;
            double[][] dWhh1;
            double[][] dWxh2;
            double[][] dWhh2;
            double[][] dWhy;
            double[] dbh1;
            double[] dbh2;
            double[] dby;

            BatchResult() {
                loss = 0;
                validPairs = 0;
                dWe = new double[vocabSize][embeddingSize];
                dWxh1 = new double[hiddenSize][embeddingSize];
                dWhh1 = new double[hiddenSize][hiddenSize];
                dWxh2 = new double[hiddenSize][hiddenSize];
                dWhh2 = new double[hiddenSize][hiddenSize];
                dWhy = new double[vocabSize][hiddenSize];
                dbh1 = new double[hiddenSize];
                dbh2 = new double[hiddenSize];
                dby = new double[vocabSize];
            }
        }

        for (int epoch = 0; epoch < epochs; epoch++) {
            List<BatchResult> batchResults = Collections.synchronizedList(new ArrayList<>());
            int steps = (words.size() - seqLength + batchSize - 1) / batchSize;
            List<Runnable> tasks = new ArrayList<>();

            for (int batchStart = 0; batchStart < words.size() - seqLength; batchStart += batchSize) {
                int batchEnd = Math.min(batchStart + batchSize, words.size() - seqLength);
                int finalBatchStart = batchStart;

                tasks.add(() -> {
                    BatchResult result = new BatchResult();

                    for (int t = finalBatchStart; t < finalBatchStart + batchSize && t < batchEnd; t++) {
                        double[][] xHistory = new double[seqLength][vocabSize];
                        double[][] yTrueHistory = new double[seqLength][vocabSize];
                        boolean validSequence = true;

                        for (int s = 0; s < seqLength; s++) {
                            if (t + s + 1 >= words.size()) {
                                validSequence = false;
                                break;
                            }
                            String currentWord = words.get(t + s);
                            String nextWord = words.get(t + s + 1);
                            if (!wordToIndex.containsKey(currentWord) || !wordToIndex.containsKey(nextWord)) {
                                validSequence = false;
                                break;
                            }
                            xHistory[s][wordToIndex.get(currentWord)] = 1.0;
                            yTrueHistory[s][wordToIndex.get(nextWord)] = 1.0;
                        }

                        if (!validSequence) continue;

                        double[] h1 = new double[hiddenSize];
                        double[] h2 = new double[hiddenSize];
                        double[][][] forwardResult = forward(xHistory, h1, h2, seqLength);
                        for (int s = 0; s < seqLength; s++) {
                            for (int i = 0; i < vocabSize; i++) {
                                result.loss += -yTrueHistory[s][i] * Math.log(forwardResult[2][s][i] + 1e-10);
                            }
                        }
                        result.validPairs += seqLength;

                        Object[] gradients = backwardParallel(xHistory, yTrueHistory, forwardResult, seqLength);
                        double[][] dWe = (double[][]) gradients[0];
                        double[][] dWxh1 = (double[][]) gradients[1];
                        double[][] dWhh1 = (double[][]) gradients[2];
                        double[][] dWxh2 = (double[][]) gradients[3];
                        double[][] dWhh2 = (double[][]) gradients[4];
                        double[][] dWhy = (double[][]) gradients[5];
                        double[] dbh1 = (double[]) gradients[6];
                        double[] dbh2 = (double[]) gradients[7];
                        double[] dby = (double[]) gradients[8];

                        for (int i = 0; i < vocabSize; i++) {
                            for (int j = 0; j < embeddingSize; j++) {
                                result.dWe[i][j] += dWe[i][j];
                            }
                            for (int j = 0; j < hiddenSize; j++) {
                                result.dWhy[i][j] += dWhy[i][j];
                            }
                        }
                        for (int i = 0; i < hiddenSize; i++) {
                            for (int j = 0; j < embeddingSize; j++) {
                                result.dWxh1[i][j] += dWxh1[i][j];
                            }
                            for (int j = 0; j < hiddenSize; j++) {
                                result.dWhh1[i][j] += dWhh1[i][j];
                                result.dWxh2[i][j] += dWxh2[i][j];
                                result.dWhh2[i][j] += dWhh2[i][j];
                            }
                            result.dbh1[i] += dbh1[i];
                            result.dbh2[i] += dbh2[i];
                        }
                        for (int i = 0; i < vocabSize; i++) {
                            result.dby[i] += dby[i];
                        }
                    }

                    batchResults.add(result);
                });
            }

            for (Runnable task : tasks) {
                executor.submit(task);
            }
            executor.awaitTermination(1, TimeUnit.HOURS);

            // Acumular resultados
            double totalLoss = 0;
            int totalValidPairs = 0;
            double[][] globalDWe = new double[vocabSize][embeddingSize];
            double[][] globalDWxh1 = new double[hiddenSize][embeddingSize];
            double[][] globalDWhh1 = new double[hiddenSize][hiddenSize];
            double[][] globalDWxh2 = new double[hiddenSize][hiddenSize];
            double[][] globalDWhh2 = new double[hiddenSize][hiddenSize];
            double[][] globalDWhy = new double[vocabSize][hiddenSize];
            double[] globalDbh1 = new double[hiddenSize];
            double[] globalDbh2 = new double[hiddenSize];
            double[] globalDby = new double[vocabSize];

            for (BatchResult result : batchResults) {
                totalLoss += result.loss;
                totalValidPairs += result.validPairs;
                for (int i = 0; i < vocabSize; i++) {
                    for (int j = 0; j < embeddingSize; j++) {
                        globalDWe[i][j] += result.dWe[i][j];
                    }
                    for (int j = 0; j < hiddenSize; j++) {
                        globalDWhy[i][j] += result.dWhy[i][j];
                    }
                }
                for (int i = 0; i < hiddenSize; i++) {
                    for (int j = 0; j < embeddingSize; j++) {
                        globalDWxh1[i][j] += result.dWxh1[i][j];
                    }
                    for (int j = 0; j < hiddenSize; j++) {
                        globalDWhh1[i][j] += result.dWhh1[i][j];
                        globalDWxh2[i][j] += result.dWxh2[i][j];
                        globalDWhh2[i][j] += result.dWhh2[i][j];
                    }
                    globalDbh1[i] += result.dbh1[i];
                    globalDbh2[i] += result.dbh2[i];
                }
                for (int i = 0; i < vocabSize; i++) {
                    globalDby[i] += result.dby[i];
                }
            }

            // Actualizar pesos con gradientes globales
            t++;
            updateAdam(We, globalDWe, adamMWe, adamVWe);
            updateAdam(Wxh1, globalDWxh1, adamM1, adamV1);
            updateAdam(Whh1, globalDWhh1, adamM2, adamV2);
            updateAdam(Wxh2, globalDWxh2, new double[hiddenSize][hiddenSize], new double[hiddenSize][hiddenSize]);
            updateAdam(Whh2, globalDWhh2, new double[hiddenSize][hiddenSize], new double[hiddenSize][hiddenSize]);
            updateAdam(Why, globalDWhy, adamMy, adamVy);
            for (int i = 0; i < hiddenSize; i++) {
                bh1[i] -= learningRate * globalDbh1[i];
                bh2[i] -= learningRate * globalDbh2[i];
            }
            for (int i = 0; i < vocabSize; i++) {
                by[i] -= learningRate * globalDby[i];
            }

            double avgLoss = totalValidPairs > 0 ? totalLoss / totalValidPairs : 0;
            System.out.printf("Epoch %d, Loss: %.4f%n", epoch, avgLoss);

            if (avgLoss > prevLoss) {
                learningRate *= 0.5;
                System.out.println("Pérdida aumentó, reduciendo learning rate a: " + learningRate);
            }
            prevLoss = avgLoss;

            if (epoch == epochs - 1) {
                saveModel("model_epoch_" + epoch + ".dat");
            }
        }
        executor.shutdown();
    }

    // Generar texto
    public String generateText(String seed, int length, double temperature) {
        StringBuilder generated = new StringBuilder(seed);
        double[] h1 = new double[hiddenSize];
        double[] h2 = new double[hiddenSize];
        String currentWord = seed.toLowerCase();

        for (int i = 0; i < length; i++) {
            double[] x = new double[vocabSize];
            x[wordToIndex.getOrDefault(currentWord, 0)] = 1.0;
            double[][][] result = forward(new double[][]{x}, h1, h2, 1);
            h1 = result[0][0];
            h2 = result[1][0];
            double[] y = result[2][0];

            for (int j = 0; j < y.length; j++) {
                y[j] = Math.pow(y[j], 1.0 / temperature);
            }
            double sum = Arrays.stream(y).sum();
            for (int j = 0; j < y.length; j++) {
                y[j] /= sum;
            }

            int nextIdx = sampleFromDistribution(y);
            currentWord = indexToWord.getOrDefault(nextIdx, "<UNK>");
            generated.append(" ").append(currentWord);
        }
        return generated.toString();
    }

    private int sampleFromDistribution(double[] dist) {
        double r = rand.nextDouble();
        double sum = 0;
        for (int i = 0; i < dist.length; i++) {
            sum += dist[i];
            if (r <= sum) return i;
        }
        return dist.length - 1;
    }

    public void saveModel(String filePath) throws IOException {
        try (ObjectOutputStream out = new ObjectOutputStream(new FileOutputStream(filePath))) {
            out.writeObject(this);
        }
        System.out.println("Modelo guardado en: " + filePath);
    }

    public static SpanishPowerPCLLM loadModel(String filePath) throws IOException, ClassNotFoundException {
        try (ObjectInputStream in = new ObjectInputStream(new FileInputStream(filePath))) {
            SpanishPowerPCLLM model = (SpanishPowerPCLLM) in.readObject();
            System.out.println("Modelo cargado desde: " + filePath);
            return model;
        }
    }

    public void startInteractiveMode() throws IOException, InterruptedException {
        Scanner scanner = new Scanner(System.in);
        System.out.println("Modo interactivo iniciado. Escribe 'EXIT' para salir, 'RECORD' para guardar la conversación, 'LOAD_TEXT' para cargar un archivo de texto.");
        while (true) {
            System.out.print("Usuario: ");
            String input = scanner.nextLine().trim();
            if (input.equalsIgnoreCase("EXIT")) {
                break;
            } else if (input.equalsIgnoreCase("RECORD")) {
                recordConversation();
                continue;
            } else if (input.equalsIgnoreCase("LOAD_TEXT")) {
                System.out.print("Ruta del archivo de texto: ");
                String filePath = scanner.nextLine().trim();
                try {
                    loadCorpus(filePath);
                    train(new ArrayList<>(wordToIndex.keySet()), 1, 128, 20);
                    System.out.println("Texto cargado y modelo entrenado.");
                } catch (IOException | InterruptedException e) {
                    System.out.println("Error al cargar el archivo: " + e.getMessage());
                }
                continue;
            }

            conversation.add("Usuario: " + input);
            String[] inputWords = input.toLowerCase().split("\\s+");
            String seed = inputWords.length > 0 ? inputWords[0] : "<UNK>";
            String response = generateText(seed, 10, 0.7);
            System.out.println("Modelo: " + response);
            conversation.add("Modelo: " + response);
        }
        scanner.close();
    }

    private void recordConversation() throws IOException, InterruptedException {
        if (conversation.isEmpty()) {
            System.out.println("No hay conversación para registrar.");
            return;
        }

        String tempFile = "conversation_temp.txt";
        try (BufferedWriter bw = new BufferedWriter(new FileWriter(tempFile))) {
            for (String line : conversation) {
                bw.write(line);
                bw.newLine();
            }
        }

        loadCorpus(tempFile);
        System.out.println("Reentrenando con la conversación...");
        train(new ArrayList<>(wordToIndex.keySet()), 1, 128, 20);
        conversation.clear();
        new File(tempFile).delete();
        System.out.println("Conversación registrada y modelo actualizado.");
    }

    public static void main(String[] args) {
        try {
            SpanishPowerPCLLM model;
            String modelPath = "model_epoch_0.dat";
            File modelFile = new File(modelPath);

            if (modelFile.exists()) {
                model = loadModel(modelPath);
            } else {
                model = new SpanishPowerPCLLM();
                model.loadCorpus("src/data/corpus.txt");
                model.train(new ArrayList<>(model.wordToIndex.keySet()), 20, 128, 20);
            }
            model.startInteractiveMode();
        } catch (IOException | ClassNotFoundException | InterruptedException e) {
            e.printStackTrace();
        }
    }
}
