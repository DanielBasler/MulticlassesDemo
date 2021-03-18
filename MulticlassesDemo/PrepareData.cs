using System;

namespace MulticlassesDemo
{
    public class PrepareData
    {
        double[][] trainingDataItems = null;
        double[][] testingDataItems = null;

        public void PreparingTraningData(double[][] allDataItems, int seed, int traingCourseData)
        {
            decimal courseData = Convert.ToDecimal(traingCourseData) / 100;
            Random rnd = new Random(seed);

            int totalRows = allDataItems.Length;
            int numberCols = allDataItems[0].Length;

            int trainingRows = (int)(totalRows * courseData);
            int testingRows = totalRows - trainingRows;

            trainingDataItems = new double[trainingRows][];
            testingDataItems = new double[testingRows][];
            double[][] copy = new double[allDataItems.Length][];

            for (int i = 0; i < copy.Length; ++i)
            {
                copy[i] = allDataItems[i];
            }                

            for (int i = 0; i < copy.Length; ++i)
            {
                int r = rnd.Next(i, copy.Length);
                double[] tmp = copy[r];
                copy[r] = copy[i];
                copy[i] = tmp;
            }           

            for (int i = 0; i < trainingRows; ++i)
            {
                trainingDataItems[i] = new double[numberCols];

                for (int j = 0; j < numberCols; ++j)
                {
                    trainingDataItems[i][j] = copy[i][j];
                }
            }            

            for (int i = 0; i < testingRows; ++i)
            {
                testingDataItems[i] = new double[numberCols];

                for (int j = 0; j < numberCols; ++j)
                {
                    testingDataItems[i][j] = copy[i + trainingRows][j];
                }
            }           
        }

        public double[][] GetItemsOfTrainingData()
        {
            return trainingDataItems;
        }

        public double[][] GetItemsOfTestingData()
        {
            return testingDataItems;
        }
    }
}

