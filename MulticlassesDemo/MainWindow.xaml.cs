using System;
using System.Collections.Generic;
using System.IO;
using System.Windows;

namespace MulticlassesDemo
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {
        NeuralNetwork neuralNetwork = null;
        NeuralMath neuralMath = new NeuralMath();
        PrepareData prepareData = new PrepareData();
        int nNeuronInputLayer = 4;
        int nNeuronOutputLayer = 3;
        int activationFunction = 1;
        int neuronOfHiddenLayer = 0;
        int trainingCourseData = 0;
        int initWeights = 0;
        int epoch = 0;
        double learnRate = 0;
        double momentumFactor = 0;
        double meanSquaredError = 0;
        double[][] allDataItems = new double[89][];
        double[][] trainingDataItems = null;
        double[][] testingDataItems = null;
        List<string> OutputText = new List<string>();


        public MainWindow()
        {
            InitializeComponent();
        }

        private void CbHyperTan_Checked(object sender, RoutedEventArgs e)
        {
            if (cbHyperTan.IsChecked == true)
            {
                cbSigmoid.IsChecked = false;
                activationFunction = 1;
            }            
        }

        private void CbSigmoid_Checked(object sender, RoutedEventArgs e)
        {
            if (cbSigmoid.IsChecked == true)
            {
                cbHyperTan.IsChecked = false;
                activationFunction = 2;
            }
        }

        private void BCancel_Click(object sender, RoutedEventArgs e)
        {
            Application.Current.Shutdown();
        }

        private void BStart_Click(object sender, RoutedEventArgs e)
        {
            OutputText.Add("Start...");
            //No error handling of missing entries
            neuronOfHiddenLayer = Convert.ToInt16(teNeuronOfHiddenLayer.Text);
            OutputText.Add("Input-Neuronen = 4");
            OutputText.Add("Hidden-Neuronen = " + teNeuronOfHiddenLayer.Text);
            OutputText.Add("Output-Neuronen = 3");
            trainingCourseData = Convert.ToInt16(teTrainingCourseData.Text);
            OutputText.Add("Traningsdatenanteil = " + teTrainingCourseData.Text + " %");
            initWeights = Convert.ToInt16(teWeights.Text);
            OutputText.Add("Initialisierungswert der Gewichte = " + teWeights.Text);
            epoch = Convert.ToInt16(teEpoche.Text);
            OutputText.Add("Epoche = " + teEpoche.Text);
            learnRate = Convert.ToDouble(teLearnRate.Text);
            OutputText.Add("Lernrate = " + teLearnRate.Text);
            meanSquaredError = Convert.ToDouble(teMeanSquaredError.Text);
            OutputText.Add("Mittlere quadratische Abweichung = " + teMeanSquaredError.Text);
            momentumFactor = Convert.ToDouble(teMomentumFactor.Text);
            OutputText.Add("Momentum-Faktor = " + teMomentumFactor.Text);

            if(activationFunction == 1)
            {
                OutputText.Add("Aktievierungsfunktion = HyperTan");
            }
            else
            {
                OutputText.Add("Aktievierungsfunktion = Sigmoid");
            }

            OutputText.Add("");
            OutputText.Add("Lese CSV Daten...");
            ListProgramValues.ItemsSource = OutputText;

            ReadingDataset();

            prepareData.PreparingTraningData(allDataItems, initWeights, trainingCourseData);
            trainingDataItems = prepareData.GetItemsOfTrainingData();
            testingDataItems = prepareData.GetItemsOfTestingData();

            neuralNetwork = new NeuralNetwork(nNeuronInputLayer, neuronOfHiddenLayer, nNeuronOutputLayer, activationFunction);            
            neuralNetwork.Training(trainingDataItems, epoch, learnRate, momentumFactor, meanSquaredError);
            OutputText.Add("");
            OutputText.Add("Training durchgeführt.");
            ListProgramValues.Items.Refresh();
            
            double trainingForecast = neuralMath.Forecast(trainingDataItems, nNeuronInputLayer, nNeuronOutputLayer);
            OutputText.Add("Genauigkeit der Trainingsdaten = " + Convert.ToString(trainingForecast));
            OutputText.Add("");
            double testingForecast = neuralMath.Forecast(testingDataItems, nNeuronInputLayer, nNeuronOutputLayer);
            OutputText.Add("Genauigkeit der Testdaten = " + Convert.ToString(testingForecast));

            OutputText.Add("");
            OutputText.Add("Berechnung beendet...");
            ListProgramValues.Items.Refresh();


        }

        private void ReadingDataset()
        {
            try
            {
                if (File.Exists(@"c://temp//waelzlager.csv"))
                {
                    int index = 0;
                    foreach (var line in File.ReadAllLines(@"c://temp//waelzlager.csv"))
                    {
                        try
                        {
                            var splittedLine = line.Split(';');

                            allDataItems[index] = new double[]
                            {
                                Convert.ToDouble(splittedLine[0]),
                                Convert.ToDouble(splittedLine[1]),
                                Convert.ToDouble(splittedLine[2]),
                                Convert.ToDouble(splittedLine[3]),
                                Convert.ToDouble(splittedLine[4]),
                                Convert.ToDouble(splittedLine[5]),
                                Convert.ToDouble(splittedLine[6])
                            };                           

                            index++;
                        }
                        catch (Exception) { } // ToDo
                    }
                }
            }
            catch (Exception e)
            {
                //ToDo
            }
        }
    }
}
