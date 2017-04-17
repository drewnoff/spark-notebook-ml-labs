# Neural Networks & Backpropagation with ND4J

In this lab we're going to implement a small framework for training neural networks for classification tasks using [ND4J](http://nd4j.org/) numerical computing library .

This lab is not intended to provide full explanation of underlying theory. Recommended materials: [deeplearningbook.org](http://www.deeplearningbook.org/), [Introduction to Deep Learning leacture slides](https://m2dsupsdlclass.github.io/lectures-labs/).

Our framework will support following neural network layers.

<img src="http://telegra.ph/file/175a34024bc45651d0be6.png" width=500>
</img>

 - **Fully-connected layer (or dense layer)**. Neurons in a fully connected layer have full connections to all activations in the previous layer. Their activations can hence be computed with a matrix multiplication followed by a bias offset.*
 
  <img src="http://telegra.ph/file/d2c25b153883ab5964ac9.png" align="center" border="0" alt="\mathrm{Dense} \equiv f\left(\textbf{x}\right)=\textbf{W}\textbf{x}+\textbf{b}" width="197" height="19" />,
 
 where
 <img src="http://telegra.ph/file/382e7cd46918ea933991c.png" align="center" border="0" alt="\textbf{W}\in\mathbb{R}^{(k,n)}" width="237" height="19" /> - weight matrix,
 <img src="http://telegra.ph/file/ba96de4353ecf27f45a2a.png" align="center" border="0" alt="\textbf{b}\in\mathbb{R}^k" width="215" height="19" /> - bias offset.
 
 
 - **Sigmoid activation layer**. 
 
    <img src="http://telegra.ph/file/c8fd772f3bab71cc82c30.png" align="center" border="0" alt="\mathrm{Sigmoid} \equiv f\left(\textbf{x}\right)=\frac{1}{1+\exp^{\textbf{-x}}}" width="229" height="46" />
    
    
 - **[Dropout layer](https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf)**. It's introduced to prevent overfitting.
 It takes parameter $d$ which is equal to probability of individual neuron being "dropped out" during the *training stage* independently for each training example. The removed nodes are then reinserted into the network with their original weights. At *testing stage* we're using the full network with each neuron's output weighted by a factor of $1-d$, so the expected value of the output of any neuron is the same as in the training stages.
 
    <img src="http://telegra.ph/file/e965e8a3ac05cbb893efe.png" align="center" border="0" alt=" $$\mathrm{Dropout_{train}} \equiv f\left(\textbf{x}\right)=\textbf{m}\odot\textbf{x}$$    $$\textbf{m} \in \left\{0,1\right\}^{n}$$    $$p\left(m_{i}=0\right)=d$$        $$\mathrm{Dropout_{test}}\equiv f\left(\textbf{x}\right)=\left(1-d\right)\textbf{x}$$" width="444" height="76" />
    
    
 - **Softmax classifier layer**. It's a generalization of binary Logistic Regression classifier to multiple classes. The Softmax classifier gives normalized class probabilities as its output.
  
  <img src="http://telegra.ph/file/110df4eb679e501246a78.png" align="center" border="0" alt=" $$\mathrm{Softmax}_{i} \equiv p_{i}\left(\textbf{x}\right)=\frac{e^{x_{i}}}{\sum_{j}{e^{x_{j}}}}$$" width="204" height="36" />
   
   We will use the Softmax classifier together with **cross-entropy loss** which is a generalization of binary log loss for multiple classes.
   The cross-entropy between a “true” distribution $p$ and an estimated distribution $q$ is defined as:
   
   <img src="http://telegra.ph/file/f8a1df033c2f8e31d790e.png" align="center" border="0" alt="$$\mathcal{L}=-\sum_{i}{p_{i}\log{q_{i}}}$$" width="137" height="22" />
   
   The Softmax classifier is hence minimizing the cross-entropy between the estimated class probabilities and the “true” distribution, where "true" distribution <img src="http://telegra.ph/file/62ef1f14d98148e90ea40.png" align="center" border="0" alt="$\textbf{p}=\left[p_{1}...p_{i}...\right]$" width="114" height="19" /> with only one element is equal to $1$ (true class) and all the other are equal to $0$. 

## Install ND4J

### Prerequisites

 - [JavaCPP](http://nd4j.org/getstarted#javacpp)
 - [BLAS (ATLAS, MKL, or OpenBLAS)](http://nd4j.org/getstarted#blas)
 
These will vary depending on whether you’re running on CPUs or GPUs.
The default backend for CPUs is `nd4j-native-platform`, and for CUDA it is `nd4j-cuda-7.5-platform`.

Assuming the default backend for CPUs is used, `customDeps` section of Spark Notebook metadata (`Edit` -> `Edit Notebook Metadata`) should look like following:

```
 "customDeps": [
    "org.bytedeco % javacpp % 1.3.2",
    "org.nd4j % nd4j-native-platform % 0.8.0",
    "org.nd4j %% nd4s % 0.8.0",
    "org.deeplearning4j % deeplearning4j-core % 0.8.0"
  ]
```

**[ND4J user guide](http://nd4j.org/userguide)** might be of the great help to track neural network components implementation.

```scala
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.ops.transforms.Transforms
import org.nd4s.Implicits._

import org.nd4j.linalg.cpu.nativecpu.rng.CpuNativeRandom
```

```scala
val rngSEED = 181
val RNG = new CpuNativeRandom(rngSEED)
```

## Sigmoid & Softmax functions 

First let's implement **`sigmoid`** and **`sigmoidGrad`** functions:

 - **`sigmoid`** function applies sigmoid transformation in an element-wise manner to each row of the input;
 - **`sigmoidGrad`** computes the gradient for the sigmoid function. It takes sigmoid function value as an input. 
 
```scala
def sigmoid(x: INDArray): INDArray = {
  Transforms.pow(Transforms.exp(-x) + 1, -1)
}
 

def sigmoidGrad(f: INDArray): INDArray = {
  f * (-f + 1)
}
```

We used [`Transform ops`](http://nd4j.org/userguide#opstransform) to apply element-wise `exp` and `pow`.

**`softmax`** computes the softmax function for each row of the input.

```scala
def softmax(x: INDArray): INDArray = {
  val exps = Transforms.exp(x.addColumnVector(-x.max(1)))
  exps.divColumnVector(exps.sum(1))
}
```

In addition to previously seen `Transforms ops` we also used [`Vector ops`](http://nd4j.org/userguide#opsbroadcast) here to subtract from each row its max element and divide each row by the sum of its elements. 

```scala
def sigmoidTest(): Unit = {
  val x = Array(Array(1, 2), Array(-1, -2)).toNDArray
  val f = sigmoid(x)
  val g = sigmoidGrad(f)
  val sigmoidVals = Array(Array(0.73105858, 0.88079708),
                          Array(0.26894142, 0.11920292)).toNDArray
  val gradVals = Array(Array(0.19661193, 0.10499359),
                       Array(0.19661193, 0.10499359)).toNDArray
  assert((f - Transforms.abs(sigmoidVals)).max(1) < 1e-6)
  assert((g - Transforms.abs(gradVals)).max(1) < 1e-6)
  println("sigmoid tests passed")
}


def softmaxTest(): Unit = {
  val x = Array(Array(1001, 1002), 
                Array(3, 4)).toNDArray
  val logits = softmax(x)
  val expectedLogits = Array(Array(0.26894142, 0.73105858),
                             Array(0.26894142, 0.73105858)).toNDArray
  assert((logits - Transforms.abs(expectedLogits)).max(1) < 1e-6)
  assert(
    (softmax(Array(1, 1).toNDArray) - Transforms.abs(Array(0.5, 0.5).toNDArray)).max(1) < 1e-6
  )
  println("softmax tests passed")
}
```

```scala
sigmoidTest
softmaxTest
```

## Network Layers

Let's define `NetLayer` trait for building network layers. We need to provide two methods:
 - `forwardProp` for forward propagation of input through the neural network in order to generate the network's output.
 - `backProp` for delta backpropagation and weights update. 
   `backProp` takes the weight's output gradients with respect to layer's inputs. The weight's output gradient and input activation are multiplied to find the gradient of the weight. A ratio (gets tuned by `learningRate`) of the weight's gradient is subtracted from the weight.

```scala
trait NetLayer {
  def forwardProp(inputs: INDArray, isTrain: Boolean): INDArray
  def backProp(outputsGrad: INDArray): INDArray
}
```

```scala
class Dense(inputDim: Int, outputDim: Int, val learningRate: Double) extends NetLayer {
  private val W = Nd4j.rand(Array(inputDim, outputDim), -0.01, 0.01, RNG)
  private val b = Nd4j.rand(Array(1, outputDim), -0.01, 0.01, RNG)
  private var _inputs = Nd4j.zeros(1, inputDim)
  
  def forwardProp(inputs: INDArray, isTrain: Boolean): INDArray = {
    _inputs = inputs
    (inputs mmul W) addRowVector b
  }
  
  def backProp(outputsGrad: INDArray): INDArray = {
    val gradW = _inputs.T mmul outputsGrad
    val gradb = outputsGrad.sum(0)
    val prop = outputsGrad mmul W.T
    W -= gradW * learningRate
    b -= gradb * learningRate
    prop
  }
}
```

```scala
class SigmoidActivation extends NetLayer {
  private var _outputs = Nd4j.zeros(1)
  
  def forwardProp(inputs: INDArray, isTrain: Boolean): INDArray = {
    _outputs = sigmoid(inputs)
    _outputs
  }
  
  def backProp(outputsGrad: INDArray): INDArray = {
    outputsGrad * sigmoidGrad(_outputs)
  }
}
```

```scala
class Dropout(val dropRate: Double = 0.0) extends NetLayer {
  var mask: INDArray = Nd4j.zeros(1)
  def forwardProp(inputs: INDArray, isTrain: Boolean): INDArray = {
    if (isTrain) {
      mask = Nd4j.zeros(1, inputs.shape()(1))
      Nd4j.choice(Array(0, 1).toNDArray, Array(dropRate, 1 - dropRate).toNDArray, mask)
      inputs.mulRowVector(mask)
    } else {
      inputs * (1 - dropRate)
    }    
  }
  
  def backProp(outputsGrad: INDArray): INDArray = {
    outputsGrad.mulRowVector(mask)
  }
}
```

We assume that the **Softmax** is always the last layer of the network.

Also it can be shown that the gradient of cross-entropy loss of the outputs of softmax layer with respect to softmax layer's input has a simple form:

<img src="http://telegra.ph/file/7d07acf33088acd4bebc4.png" align="center" border="0" alt=" $$\frac{\partial \mathcal{L}}{\partial x_{i}}=g_{i}-p_{i}$$" width="97" height="28" />
 
 So to start backpropagation stage let's take the `Softmax` output probabilities alongside with true labels as an input for `backProp` method of the `Softmax` layer.


```scala
import org.nd4s.Implicits._

class Softmax extends NetLayer {
  def forwardProp(inputs: INDArray, isTrain: Boolean): INDArray = {
    softmax(inputs)
  }
  
  def backProp(outputsGrad: INDArray): INDArray = {
    val predictions = outputsGrad(0, ->)
    val labels = outputsGrad(1, ->)
    predictions - labels
  }
}
```

```scala
def crossEntropy(predictions: INDArray, labels: INDArray): Double = {
  val cost = - (Transforms.log(predictions) * labels).sumNumber.asInstanceOf[Double]
  cost / labels.shape()(0)
}
```

```scala
def accuracy(predictions: INDArray, labels: INDArray): Double = {
  val samplesNum = labels.shape()(0)
  val matchesNum = (Nd4j.argMax(predictions, 1) eq Nd4j.argMax(labels, 1)).sumNumber.asInstanceOf[Double]
  100.0 * matchesNum / samplesNum
}
```

## Neural Network

```scala
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import org.nd4j.linalg.dataset.DataSet
```

```scala
case class Metric(epoch: Int, acc: Double, loss: Double)
```

We will use the class called `DataSetIterator` to fetch `DataSet`s. 

```scala
import scala.collection.JavaConverters._


case class NeuralNet(layers: Vector[NetLayer] = Vector()) {
  
  def addLayer(layer: NetLayer): NeuralNet = {
    this.copy(layers :+ layer)
  }
  
  def fit(trainData: DataSetIterator, numEpochs: Int, validationData: DataSet): Seq[Metric] = {
    val history = (1 to numEpochs).foldLeft(List[Metric]()){ (history, epoch) =>
      trainData.reset()
      trainData.asScala.foreach ( ds => trainBatch(ds.getFeatures, ds.getLabels) )
      
      // validate on validation Dataset
      val prediction = this.predict(validationData.getFeatures)
      val loss = crossEntropy(prediction, validationData.getLabels)
      val acc = accuracy(prediction, validationData.getLabels)
      
      println(s"Epoch: $epoch/$numEpochs - loss: $loss - acc: $acc")

      Metric(epoch, acc, loss) :: history
    }
    history.reverse
  }
  
  def predict(X: INDArray): INDArray = {
    layers.foldLeft(X){
      (input, layer) => layer.forwardProp(input, isTrain=false)
    }
  }
    
  private def trainBatch(X: INDArray, Y: INDArray): Unit = {
    val YPredict = layers.foldLeft(X){
      (input, layer) => layer.forwardProp(input, isTrain=true)
    }
    val shape = Y.shape
    layers.reverse.foldLeft(
      Nd4j.vstack(YPredict, Y).reshape(2, shape(0), shape(1))
    ){
      (deriv, layer) => layer.backProp(deriv)
    }
  }  
}
```

## MNIST

Now let's apply our framework to build neural network for MNIST dataset classification.
The `DatasetIterator` implementation called `MnistDataSetIterator` is available in `deeplearning4j` to iterate over MNIST dataset.

```scala
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator
```

```scala
val learningRate = 0.01
val batchSize = 128

val mnistTrain = new MnistDataSetIterator(batchSize, true, rngSEED)
val mnistTest = new MnistDataSetIterator(batchSize, false, rngSEED)

val inputDim = mnistTest.next.getFeatures.shape()(1)
val totalTestExamples = mnistTest.numExamples()
```

```scala
val model = NeuralNet()
            .addLayer(new Dense(inputDim=inputDim, outputDim=512, learningRate=learningRate))
            .addLayer(new SigmoidActivation())
            .addLayer(new Dropout(dropRate=0.3))
            .addLayer(new Dense(512, 512, learningRate))
            .addLayer(new SigmoidActivation())
            .addLayer(new Dropout(0.3))
            .addLayer(new Dense(512, 10, learningRate))
            .addLayer(new Softmax())
```

```scala
val history = model.fit(mnistTrain, 40, (new MnistDataSetIterator(totalTestExamples, false, rngSEED)).next)
```
> Epoch: 1/40 - loss: 0.57162705078125 - acc: 81.94
Epoch: 2/40 - loss: 0.348628173828125 - acc: 89.34
Epoch: 3/40 - loss: 0.273960546875 - acc: 91.83
Epoch: 4/40 - loss: 0.2305306396484375 - acc: 92.76
Epoch: 5/40 - loss: 0.20194395751953126 - acc: 93.76
Epoch: 6/40 - loss: 0.17214320068359376 - acc: 94.87
Epoch: 7/40 - loss: 0.15777041015625 - acc: 95.29
Epoch: 8/40 - loss: 0.1411923583984375 - acc: 95.75
Epoch: 9/40 - loss: 0.1371442138671875 - acc: 95.65
Epoch: 10/40 - loss: 0.1223932373046875 - acc: 96.2
Epoch: 11/40 - loss: 0.11889525146484375 - acc: 96.35
Epoch: 12/40 - loss: 0.11355523681640625 - acc: 96.5
Epoch: 13/40 - loss: 0.10255557861328125 - acc: 96.63
Epoch: 14/40 - loss: 0.10248739013671875 - acc: 96.67
Epoch: 15/40 - loss: 0.10121082153320313 - acc: 96.76
Epoch: 16/40 - loss: 0.09314661254882813 - acc: 97.05
Epoch: 17/40 - loss: 0.0908234619140625 - acc: 97.09
Epoch: 18/40 - loss: 0.08782809448242188 - acc: 97.21
Epoch: 19/40 - loss: 0.084460498046875 - acc: 97.25
Epoch: 20/40 - loss: 0.08508148803710938 - acc: 97.32
Epoch: 21/40 - loss: 0.08242890625 - acc: 97.49
Epoch: 22/40 - loss: 0.07931015014648438 - acc: 97.55
Epoch: 23/40 - loss: 0.07825602416992188 - acc: 97.6
Epoch: 24/40 - loss: 0.07847127685546874 - acc: 97.47
Epoch: 25/40 - loss: 0.07547276611328126 - acc: 97.6
Epoch: 26/40 - loss: 0.074110009765625 - acc: 97.64
Epoch: 27/40 - loss: 0.07486264038085938 - acc: 97.69
Epoch: 28/40 - loss: 0.07151276245117187 - acc: 97.73
Epoch: 29/40 - loss: 0.07469411010742187 - acc: 97.76
Epoch: 30/40 - loss: 0.06966272583007813 - acc: 97.88
Epoch: 31/40 - loss: 0.066982666015625 - acc: 97.84
Epoch: 32/40 - loss: 0.06796741333007812 - acc: 97.87
Epoch: 33/40 - loss: 0.06789564208984375 - acc: 97.95
Epoch: 34/40 - loss: 0.065538916015625 - acc: 98.03
Epoch: 35/40 - loss: 0.066549365234375 - acc: 97.88
Epoch: 36/40 - loss: 0.06736263427734375 - acc: 97.83
Epoch: 37/40 - loss: 0.0646685302734375 - acc: 97.98
Epoch: 38/40 - loss: 0.0628564208984375 - acc: 97.97
Epoch: 39/40 - loss: 0.0657330322265625 - acc: 98.0
Epoch: 40/40 - loss: 0.063365771484375 - acc: 97.98

```scala
CustomPlotlyChart(history,
                  layout="{title: 'Accuracy on validation set', xaxis: {title: 'epoch'}, yaxis: {title: '%'}}",
                  dataOptions="{mode: 'lines'}",
                  dataSources="{x: 'epoch', y: 'acc'}")
```

<img src="http://telegra.ph/file/df9016b6205a2f69685c8.png" width=900>
</img>

```scala
CustomPlotlyChart(history,
                  layout="{title: 'Cross entropy on validation set', xaxis: {title: 'epoch'}, yaxis: {title: 'loss'}}",
                  dataOptions="""{
                    mode: 'lines', 
                    line: {
                          color: 'green',
                          width: 3
                          }
                    }""",
                  dataSources="{x: 'epoch', y: 'loss'}")
```

<img src="http://telegra.ph/file/d6c6553512a5d991e5f37.png" width=900>
</img>

## On your own:
 - Implement [rectified linear unit (ReLU)](https://en.wikipedia.org/wiki/Rectifier_(neural_networks)) activation.
 - Add support for *L2* regularization on `Dense` layer weights.
 - Train similar neural network with `relu` activation instead of `sigmoid` activation and added support for `L2` regularization. Compare obtained results.
