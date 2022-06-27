using System.Collections.Generic;
using UnityEngine;
using MathNet.Numerics.LinearAlgebra;
using System;
using Random = UnityEngine.Random;


public class NeuralNetwork : MonoBehaviour
{
    public Matrix<float> _inputLayer = Matrix<float>.Build.Dense(1, 3);

    public List<Matrix<float>> _hiddenLayers = new List<Matrix<float>>();

    public Matrix<float> _outputLayer = Matrix<float>.Build.Dense(1, 2);

    public List<Matrix<float>> _weights = new List<Matrix<float>>();

    [HideInInspector]
    public List<float> _biases = new List<float>();

    [HideInInspector]
    public float _fitness;



    public void Initialise(int hiddenLayerCount, int hiddenNeuronCount)
    {
        //CLEAR EVERYTHING
        _inputLayer.Clear();
        _hiddenLayers.Clear();
        _outputLayer.Clear();
        _weights.Clear();
        _biases.Clear();


        for (int i = 0; i < hiddenLayerCount; i++)
        {
            //ADD HIDDEN NEURONS TO HIDDEN LAYERS
            Matrix<float> f = Matrix<float>.Build.Dense(1, hiddenNeuronCount);
            _hiddenLayers.Add(f);

            //ADD BIASES
            _biases.Add(Random.Range(-1f, 1f));

            //ADD WEIGHTS FROM INPUT LAYER TO FIRST HIDDEN LAYER 
            if (i == 0)
            {
                Matrix<float> inputToH1 = Matrix<float>.Build.Dense(3, hiddenNeuronCount);
                _weights.Add(inputToH1);
            }
            else
            {
                //ADD WEIGHTS FROM HIDDEN LAYER TO HIDDEN LAYER
                Matrix<float> HiddenToHidden = Matrix<float>.Build.Dense(hiddenNeuronCount, hiddenNeuronCount);
                _weights.Add(HiddenToHidden);
            }
        }

        //ADD WEIGHT FROM HIDDEN LAYER TO OUTPUT 
        Matrix<float> OutputWeight = Matrix<float>.Build.Dense(hiddenNeuronCount, 2);
        _weights.Add(OutputWeight);
        _biases.Add(Random.Range(-1f, 1f));

        //RANDOMISE WEIGHTS
        RandomiseWeights();
    }



    public void RandomiseWeights()
    {
        //ITERATE THROUGH LIST OF MATRICES WITH WEIGHTS
        for (int i = 0; i < _weights.Count; i++)
        {
            //ITERATE THROUGH ROWS OF WEIGHT MATRIX
            for (int x = 0; x < _weights[i].RowCount; x++)
            {
                //ITERATE THROUGH COLUMNS OF WEIGHT MATRIX
                for (int y = 0; y < _weights[i].ColumnCount; y++)
                {
                    //FILL WEIGHTS WITH RANDOM NUMBERS
                    _weights[i][x, y] = Random.Range(-1f, 1f);
                }
            }
        }
    }



    public NeuralNetwork InitialiseCopy(int hiddenLayerCount, int hiddenNeuronCount)
    {
        //CREATE COPY OF NEURALNETWORK
        NeuralNetwork n = new NeuralNetwork();

        n._weights = _weights;
        n._biases = _biases;

        n.InitialiseHidden(hiddenLayerCount, hiddenNeuronCount);

        return n;
    }



    public void InitialiseHidden(int hiddenLayerCount, int hiddenNeuronCount)
    {
        //CLEAN ALL THE LAYERS
        _inputLayer.Clear();
        _hiddenLayers.Clear();
        _outputLayer.Clear();


        //ADD HIDDEN LAYERS TO COPY OF NEURAL NETWORK
        for (int i = 0; i < hiddenLayerCount + 1; i++)
        {
            Matrix<float> newHiddenLayer = Matrix<float>.Build.Dense(1, hiddenNeuronCount);
            _hiddenLayers.Add(newHiddenLayer);
        }
    }



    public (float, float) RunNetwork(float a, float b, float c)
    {
        //APPLY SENSORS AS INPUT LAYER
        _inputLayer[0, 0] = a;
        _inputLayer[0, 1] = b;
        _inputLayer[0, 2] = c;

        //APPLY TANH TO INPUTLAYER
        _inputLayer = _inputLayer.PointwiseTanh();

        //CALCULATE NEURONS OF THE FIRST HIDDEN LAYER
        _hiddenLayers[0] = ((_inputLayer * _weights[0]) + _biases[0]).PointwiseTanh();

        //CALCULATE NEURONS OF THE REST OF HIDDEN LAYERS
        for (int i = 1; i < _hiddenLayers.Count; i++)
        {
            _hiddenLayers[i] = ((_hiddenLayers[i - 1] * _weights[i]) + _biases[i]).PointwiseTanh();
        }

        //CALCULATE OUTPUT LAYER
        _outputLayer = ((_hiddenLayers[_hiddenLayers.Count - 1] * _weights[_weights.Count - 1]) + _biases[_biases.Count - 1]).PointwiseTanh();

        //FIRST OUTPUT IS ACCELERATION AND SECOND IS STEERING, SIGMOID USED
          //FOR ACCELERATION AS CAR ALWAYS MOVES FORWARD WITH DIFFERENT SPEED AND NOT BACKWARDS
          //TANH FOR STEERING AS CAR ROTATES TO RIGHT AND LEFT
        return (Sigmoid(_outputLayer[0, 0]), (float)Math.Tanh(_outputLayer[0, 1]));
    }



    private float Sigmoid(float s)
    {
        return (1 / (1 + Mathf.Exp(-s)));
    }

}