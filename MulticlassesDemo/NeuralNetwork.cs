using System;

namespace MulticlassesDemo
{

    public class NeuralNetwork
    {
        NeuralMath neuralMath = new NeuralMath();
        private static Random rnd;

        public static int activationFunction; 

        public static int inputNodes;
        public static int hiddenNodes;
        public static int outputNodes;

        public static double[] inputs;
        public static double[][] inputToHiddenWeights;

        public static double[] hiddenBiases;
        public static double[] hiddenOutputs;
        public static double[][] hiddenToOutputWeights;

        public static double[] outputBiases;
        public static double[] outputs;
        public static double[] outputGradients;

        public static double[] hiddenGradients;
        public static double[][] inputHiddenPrevWeightsDelta;
        public static double[] hiddenPrevBiasesDelta;
        public static double[][] hiddenOutputPrevWeightsDelta;
        public static double[] outputPrevBiasesDelta;        

        public NeuralNetwork(int nInput, int nHidden, int nOutput, int act)
        {
            rnd = new Random(0);
            activationFunction = act;

            inputNodes = nInput;
            hiddenNodes = nHidden;
            outputNodes = nOutput;

            inputs = new double[inputNodes];
            inputToHiddenWeights = MakeMatrikForNeuralNetwork.CreateMatrix(inputNodes, hiddenNodes);

            hiddenBiases = new double[hiddenNodes];
            hiddenOutputs = new double[hiddenNodes];
            hiddenToOutputWeights = MakeMatrikForNeuralNetwork.CreateMatrix(hiddenNodes, outputNodes);

            outputBiases = new double[outputNodes];
            outputs = new double[outputNodes];

            InitializeWeights();

            hiddenGradients = new double[hiddenNodes];
            outputGradients = new double[outputNodes];
            inputHiddenPrevWeightsDelta = MakeMatrikForNeuralNetwork.CreateMatrix(inputNodes, hiddenNodes);

            hiddenPrevBiasesDelta = new double[nHidden];
            hiddenOutputPrevWeightsDelta = MakeMatrikForNeuralNetwork.CreateMatrix(hiddenNodes, outputNodes);

            outputPrevBiasesDelta = new double[outputNodes];
        }

        private void InitializeWeights()
        {
            int numberWeights = (inputNodes * hiddenNodes) + (hiddenNodes * outputNodes) + hiddenNodes + outputNodes;

            double[] initialWeights = new double[numberWeights];
            double lo = -0.01;
            double hi = 0.01;

            for (int i = 0; i < initialWeights.Length; ++i)
            {
                initialWeights[i] = (hi - lo) * rnd.NextDouble() + lo;
            }                

            SetWeights(initialWeights);
        }

        private void SetWeights(double[] initialWeights)
        {
            int numberWeights = (inputNodes * hiddenNodes) + (hiddenNodes * outputNodes) + hiddenNodes + outputNodes;

            if (initialWeights.Length != numberWeights) throw new Exception("Fehlerhafte Array-Länge!");

            int z = 0;

            for (int i = 0; i < inputNodes; ++i)
            {
                for (int j = 0; j < hiddenNodes; ++j)
                {
                    inputToHiddenWeights[i][j] = initialWeights[z++];
                }                    
            }                

            for (int i = 0; i < hiddenNodes; ++i)
            {
                hiddenBiases[i] = initialWeights[z++];
            }                

            for (int i = 0; i < hiddenNodes; ++i)
            {
                for (int j = 0; j < outputNodes; ++j)
                {
                    hiddenToOutputWeights[i][j] = initialWeights[z++];
                }                    
            }                

            for (int i = 0; i < outputNodes; ++i)
            {
                outputBiases[i] = initialWeights[z++];
            }                
        }

        public void Training(double[][] trainingDataItems, int maxEpochs, double learnRate, double momentumFactor, double mSquaredError)
        {            
            int epoch = 0;

            double[] xValues = new double[inputNodes];
            double[] tValues = new double[outputNodes];

            int[] sequence = new int[trainingDataItems.Length];

            for (int i = 0; i < sequence.Length; ++i)
            {
                sequence[i] = i;
            }                

            while (epoch < maxEpochs)
            {
                double mse = neuralMath.MeanSquaredError(trainingDataItems, inputNodes, hiddenNodes, outputNodes);
                if (mse < mSquaredError) break;

                RandomPlay(sequence);

                for (int i = 0; i < trainingDataItems.Length; ++i)
                {
                    int idx = sequence[i];
                    Array.Copy(trainingDataItems[idx], xValues, inputNodes);
                    Array.Copy(trainingDataItems[idx], inputNodes, tValues, 0, outputNodes);

                    neuralMath.CalculateExpenses(xValues);
                    UpdateWeights(tValues, learnRate, momentumFactor);
                }

                ++epoch;
            }
        }

        private void UpdateWeights(double[] tValues, double learnRate, double momentum)
        {
            if (tValues.Length != outputNodes)
            {
                throw new Exception("Zielwerte haben nicht die gleiche Länge wie die Output-Neuronen in der Methode UpdateWeights");
            }

            for (int i = 0; i < outputNodes; ++i)
            {
                double derivative = (1 - outputs[i]) * outputs[i];
                outputGradients[i] = derivative * (tValues[i] - outputs[i]);
            }            

            for (int i = 0; i < hiddenNodes; ++i)
            {
                double derivative = (1 - hiddenOutputs[i]) * (1 + hiddenOutputs[i]);
                double sum = 0.0;

                for (int j = 0; j < outputNodes; ++j)
                {
                    double x = outputGradients[j] * hiddenToOutputWeights[i][j];
                    sum += x;
                }

                hiddenGradients[i] = derivative * sum;
            }
                        
            for (int i = 0; i < inputNodes; ++i)
            {
                for (int j = 0; j < hiddenNodes; ++j)
                {
                    double delta = learnRate * hiddenGradients[j] * inputs[i];
                    inputToHiddenWeights[i][j] += delta;                

                    inputToHiddenWeights[i][j] += momentum * inputHiddenPrevWeightsDelta[i][j];

                    inputHiddenPrevWeightsDelta[i][j] = delta;
                }
            }
            
            for (int i = 0; i < hiddenNodes; ++i)
            {
                double delta = learnRate * hiddenGradients[i];

                hiddenBiases[i] += delta;
                hiddenBiases[i] += momentum * hiddenPrevBiasesDelta[i];

                hiddenPrevBiasesDelta[i] = delta;
            }

            for (int i = 0; i < hiddenNodes; ++i)
            {
                for (int j = 0; j < outputNodes; ++j)
                {
                    double delta = learnRate * outputGradients[j] * hiddenOutputs[i];

                    hiddenToOutputWeights[i][j] += delta;

                    hiddenToOutputWeights[i][j] += momentum * hiddenOutputPrevWeightsDelta[i][j];

                    hiddenOutputPrevWeightsDelta[i][j] = delta;
                }
            }
            
            for (int i = 0; i < outputNodes; ++i)
            {
                double delta = learnRate * outputGradients[i] * 1.0;

                outputBiases[i] += delta;

                outputBiases[i] += momentum * outputPrevBiasesDelta[i];

                outputPrevBiasesDelta[i] = delta;
            }
        }

        private void RandomPlay(int[] sequence)
        {
            for (int i = 0; i < sequence.Length; ++i)
            {
                int r = rnd.Next(i, sequence.Length);
                int tmp = sequence[r];
                sequence[r] = sequence[i];
                sequence[i] = tmp;
            }
        }
    }
}
