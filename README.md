# JavaRNN-LLM
<p>
  Una RNN escrita en Java puro para competir con Transformers.<br>
  Este proyecto surge por la curiosidad de constatar si es posible crear un modelo de inteligencia en Java puro funcional que pueda competir con los modelos actuales que utilizan Transformer. El primer paso está centrado en generar texto coherente y gramaticalmente correcto en Español, una vez alcansado este propósito, el modelo base resultante podría ser "entrenado" para "enseñarle" nuevo conocimiento.
</p>

# Diseño general actual
* Arquitectura con 2 capas recurrentes (h1, h2) tipo Elman con atención pseudo-self-attention-like al estilo RNN-Attention.
* Embeddings aprendibles (We) y paso explícito por capas ocultas.
* Implementación de Adam desde cero, (<b>respetando la fórmula oficial</b>)
* Regularización con dropout aplicada manualmente.
* Modo interactivo.
* Mecanismo de reentrenamiento conversacional por medio de comandos: <b>LOAD_TEXT</b> y <b>RECORD</b>

# Licencia MIT
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
