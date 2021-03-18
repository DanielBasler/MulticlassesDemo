using System;

namespace MulticlassesDemo
{
    public class NeuralMath
    {
        public double MeanSquaredError(double[][] trainingDataItems, int inputNodes, int hiddenNodes, int outputNodes)
        {
            double sumSquaredError = 0.0;
            double[] xValues = new double[inputNodes];
            double[] tValues = new double[outputNodes];

            for (int i = 0; i < trainingDataItems.Length; ++i)
            {
                Array.Copy(trainingDataItems[i], xValues, inputNodes);
                Array.Copy(trainingDataItems[i], inputNodes, tValues, 0, outputNodes);
                double[] yValues = CalculateExpenses(xValues);

                for (int j = 0; j < outputNodes; ++j)
                {
                    double err = tValues[j] - yValues[j];
                    sumSquaredError += err * err;
                }
            }

            return sumSquaredError / trainingDataItems.Length;
        }

        public double[] CalculateExpenses(double[] xValues)
        {
            if (xValues.Length != NeuralNetwork.inputNodes)
            {
                throw new Exception("xValues Array haben eine fehlerhafte Länge.");
            }                

            double[] hSums = new double[NeuralNetwork.hiddenNodes];

            double[] oSums = new double[NeuralNetwork.outputNodes];

            for (int i = 0; i < xValues.Length; ++i)
            {
                NeuralNetwork.inputs[i] = xValues[i];
            }                

            for (int j = 0; j < NeuralNetwork.hiddenNodes; ++j)
            {
                for (int i = 0; i < NeuralNetwork.inputNodes; ++i)
                {
                    hSums[j] += NeuralNetwork.inputs[i] * NeuralNetwork.inputToHiddenWeights[i][j];
                }                    
            }                

            for (int i = 0; i < NeuralNetwork.hiddenNodes; ++i)
            {
                hSums[i] += NeuralNetwork.hiddenBiases[i];
            }

            for (int i = 0; i < NeuralNetwork.hiddenNodes; ++i)
            {
                if(NeuralNetwork.activationFunction == 1)
                {
                    NeuralNetwork.hiddenOutputs[i] = HyperTan(hSums[i]);
                }
                else
                {
                    NeuralNetwork.hiddenOutputs[i] = sigmoid(hSums[i]);
                }                
            }

            for (int j = 0; j < NeuralNetwork.outputNodes; ++j)
            {
                for (int i = 0; i < NeuralNetwork.hiddenNodes; ++i)
                {
                    oSums[j] += NeuralNetwork.hiddenOutputs[i] * NeuralNetwork.hiddenToOutputWeights[i][j];
                }                    
            }                

            for (int i = 0; i < NeuralNetwork.outputNodes; ++i)
            {
                oSums[i] += NeuralNetwork.outputBiases[i];
            }                

            double[] softOut = Softmax(oSums);
            Array.Copy(softOut, NeuralNetwork.outputs, softOut.Length);
            double[] retResult = new double[NeuralNetwork.outputNodes];
            Array.Copy(NeuralNetwork.outputs, retResult, retResult.Length);

            return retResult;
        }

        private double[] Softmax(double[] oSums)
        {
            double max = oSums[0];

            for (int i = 0; i < oSums.Length; ++i)
            {
                if (oSums[i] > max) max = oSums[i];
            }            

            double scale = 0.0;

            for (int i = 0; i < oSums.Length; ++i)
            {
                scale += Math.Exp(oSums[i] - max);
            }

            double[] result = new double[oSums.Length];

            for (int i = 0; i < oSums.Length; ++i)
            {
                result[i] = Math.Exp(oSums[i] - max) / scale;
            }               

            return result;
        }

        private double HyperTan(double activation)
        {
            if (activation < -20.0)
            {
                return -1.0;
            }
            else if (activation > 20.0)
            {
                return 1.0;
            }
            else return
                    Math.Tanh(activation);
        }

        private static double sigmoid(double activation)
        {
            return 1 / (1 + Math.Exp(-activation));
        }

        public double Forecast(double[][] testingDataItems, int inputNode, int outputNode)
        {
            int correct = 0;
            int incorrect = 0;

            double[] xValues = new double[inputNode];
            double[] tValues = new double[outputNode];
            double[] yValues;

            for (int i = 0; i < testingDataItems.Length; ++i)
            {
                Array.Copy(testingDataItems[i], xValues, inputNode);
                Array.Copy(testingDataItems[i], inputNode, tValues, 0, outputNode);

                yValues = CalculateExpenses(xValues);
                int maxIndex = MaxIndex(yValues);

                if (tValues[maxIndex] == 1.0)
                {
                    ++correct;
                }
                else
                {
                    ++incorrect;
                }
            }

            return (correct * 1.0) / (correct + incorrect);
        }

        public int MaxIndex(double[] yValues)
        {
            int bigIndex = 0;
            double biggestVal = yValues[0];

            for (int i = 0; i < yValues.Length; ++i)
            {
                if (yValues[i] > biggestVal)
                {
                    biggestVal = yValues[i];
                    bigIndex = i;
                }
            }

            return bigIndex;
        }
    }
}
