# JavaRNN-LLM
<p>
  Una RNN escrita en Java puro para competir con Transformers.<br>
  Este proyecto surge por la curiosidad de constatar si es posible crear un modelo de inteligencia en Java puro funcional que pueda competir con los modelos actuales que utilizan Transformer. El primer paso está centrado en generar texto coherente y gramaticalmente correcto en Español, una vez alcansado este propósito, el modelo base resultante podría ser "entrenado" para "enseñarle" nuevo conocimiento.
</p>

# Especificaciones del equipo con el que fue entrenado la RNN
* Intel(R) Xeon (R) CPU E5-2650 v4 @2.20GHz, 12 núcleos 24 hilos.
* NVIDIA GeForce GTX 1060 6GB VRAM
* 32GB de RAM 2133MHz

# Funcionamiento
Tamaño de vocabulario establecido por defecto: 10000 tokens <br>
Cantidad de épocas de entrenamiento inicial: 10 <br>
Tiempo de entrenamiento actual del modelo: 10h 34 min <br>
<!--Pérdida desde época inicial hasta final: Epoch 0, Loss: 9.2103 a Epoch 10, Loss: 9.2097--> <br>
Recibe un corpus en texto plano con el que se realiza el entrenamiento, para el contexto de esta implementación, se utilizó el texto extraido del Proyecto Gutenberg <img width="129" height="80" alt="pg-logo-129x80" src="https://github.com/user-attachments/assets/e8d52d69-8216-4abd-ba64-615d98acf85c" />
 <a href="https://www.gutenberg.org/cache/epub/2000/pg2000.txt">El ingenioso hidalgo don Quijote de la Mancha</a> <br>
 La implementación actual permite operar con el máximo de recursos de su computadora personal o dispositivo móvil (con ajustes previos). <br>
 Actualmente debe modificar los parámetros para ajustar el comportamiento y aprovechar los recursos de su equipo.<br>
 <b>Esta implementación está en fase de optimización por lo que se encuentra en desarrollo activo en este momento.</b>

# Diseño general actual
* Arquitectura con 2 capas recurrentes (h1, h2) tipo Elman con atención pseudo-self-attention-like al estilo RNN-Attention.
* Embeddings aprendibles (We) y paso explícito por capas ocultas.
* Implementación de Adam desde cero, (<b>respetando la fórmula oficial</b>)
* Regularización con dropout aplicada manualmente.
* Modo interactivo.
* Mecanismo de reentrenamiento conversacional por medio de comandos: <b>LOAD_TEXT</b> y <b>RECORD</b>

# Propuesta de mejoras
* Utilizar la librería Java Binding for OpenCL (JOCL) para permitir el uso de GPU Nvidia o similares en los cálculos y mejorar el desempeño de la RNN durante su entrenamiento (fine-tuning).
* Aumentar el número de capas (si con ello se logra mejora sustancial en rendimiento).
* Optimizar el mecanismo de atención (Self-attention mechanism) como lo hacen los modelos actuales para imitar la atención cognitiva, actualmente muy rudimentario.
* Usar ReLU o Swish en capas superiores para no usar Math.tanh().
* Utilizar LayerNorm o BatchNorm (si existe una implementación libre de librerías externas) para mantener el código puro.
* Implementar word2vec o fastText embedding preentrenados en español para incializar We en conjunto con one-hot + embeddings.
* Implementar UI que permita cambiar los parámetros de ejecución y sea más amigable con el usuario.
* Factibilidad de guardado del modelo después de cada época transcurrida o esperar a que acabe las determinadas.
* Validar si el uso de double en vez de float es correcto para cuestiones de manejo de memoria y escalabilidad.
