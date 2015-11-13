#include <iostream>
#include <algorithm>
#include <cstdint>
#include <cassert>
#include <vector>
#include <array>
#include <random>
#include <memory>
#include <set>

// utility class
void VectorRandomInitialize(std::vector<float>& input)
{
    // initialize to random floats in range [0,1]
    std::uniform_real_distribution<float> distribution(0.0f, 1.0f); //Values between 0 and 1
    std::mt19937 engine; // Mersenne twister MT19937
    auto generator = std::bind(distribution, engine);
    std::generate_n(input.begin(), input.size(), generator); 
}

///////////////////////////////////////////////////
// Layer Implementations
// inputDimension - number of neurons in previous layer
// outputDimension - number of neurons in the current layer
//////////////////////////////////////////////////

// Base Layer that all layers should inherit
class BaseLayer
{
public:
    BaseLayer(
        int32_t inputDim, 
        int32_t outputDim)
        : _inputDim(inputDim),
        _outputDim(outputDim)
    {}

    virtual void initializeWeights() = 0;
    virtual void forwardProp(std::vector<float>& input, std::vector<float>& output) = 0;
    virtual void backProp() = 0;

    int32_t InputDim() { return _inputDim; }
    int32_t OutputDim() { return _outputDim; }

protected:
    std::vector<float> _weights;
    int32_t _inputDim;
    int32_t _outputDim;
};

// For InputLayer, inputDim = outputDim
// and there are no weights shared.
class InputLayer : public BaseLayer
{
public:

    InputLayer(int32_t inputDim)
        : BaseLayer(inputDim, inputDim)
    {}

    void initializeWeights()
    {    
        // Nothing to Do for the input layer.
    }

    // simply return the input as output.
    void forwardProp(std::vector<float>& input, std::vector<float>& output)
    {
        std::cout << "forward prop from Input Layer" << std::endl;
        output.resize(input.size());
        // copy input to output, as the input layer does not apply any transformation.
        std::copy(input.begin(), input.end(), output.begin());
    }

    void backProp()
    {

    }
};

class OutputLayer : public BaseLayer
{
public:

    OutputLayer(int32_t inputDim, int32_t outputDim)
        : BaseLayer(inputDim, outputDim)
    {}

    void initializeWeights()
    {
        _weights.reserve(_inputDim * _outputDim);
        VectorRandomInitialize(_weights);
    }

    void forwardProp(std::vector<float>& input, std::vector<float>& output)
    {
        std::cout << "forward prop from Output Layer" << std::endl;
    }

    void backProp()
    {

    }
};

class FullyConnectedHiddenLayer : public BaseLayer
{
public:

    FullyConnectedHiddenLayer(
        int32_t inputDim, 
        int32_t outputDim)
        : BaseLayer(inputDim, outputDim)
    {}
 
    void initializeWeights()
    {
        _weights.reserve(_inputDim * _outputDim);
        VectorRandomInitialize(_weights);
    }
    
    void forwardProp(std::vector<float>& input, std::vector<float>& output)
    {
        std::cout << "forward prop from Fully Connected Layer" << std::endl;
        // perform forward propagation
        output.resize(_outputDim);
        for (int i = 0; i < input.size(); i++)
        {
            // for ith neuron, perform a dot product with all the weights.   
             
        }
    }

    void backProp()
    {

    }
};

typedef std::vector<std::shared_ptr<BaseLayer>> LayerSet;

////////////////////////////////////////
// Input Data and Data Source Related Stuff
////////////////////////////////////////

struct InputData
{
    std::vector<float> _input;
    std::vector<float> _target;
};

// source for the input data to neural network
// This is a generic class that exposes an interface to fetch input sample
// one by one. Concrete implementations can be backed by either a database, or a static dataset
class IDataFeed
{
public:
    virtual bool getNext(InputData& input) = 0;   
};

class StaticDataFeed : public IDataFeed
{
public:

    StaticDataFeed(std::vector<InputData> dataset)
        : _dataset(dataset),
        _currentOffset(0)
    {
        std::cout << "dataset size: " << _dataset.size() << "   " << _dataset[0]._input.size() << std::endl;
    }

    bool getNext(InputData& input)
    {
        if (_currentOffset < _dataset.size())
        {
            input =  _dataset[_currentOffset++];   
            return true;
        }

        return false;
    }
    
private:
    std::vector<InputData> _dataset;
    int32_t _currentOffset;
};


/////////////////////////////////////////////
// Trainer - This guy does the actual training
////////////////////////////////////////////
class Trainer
{   
public:
        
    Trainer(
        std::shared_ptr<LayerSet> layerSet, 
        std::shared_ptr<IDataFeed> dataFeed
    ) : _layers(layerSet),
    _dataFeed(dataFeed)
    {
        validate();
    }
 
    void validate()
    {
        std::cout << "Validating Layer Set" << std::endl;
        // ensure that there are atleast two layers.
        assert(_layers->size() >= 2); 

        int32_t prevLayerSize = (*_layers)[0]->InputDim();

        for (auto layer : *_layers)
        {
            assert(prevLayerSize == layer->InputDim());
            prevLayerSize = layer->OutputDim();
        }
    }

    void train()
    {
        InputData input;
        while(_dataFeed->getNext(input))
        {
            forwardProp(input);
        }
    }
    
    void forwardProp(InputData& input)
    {
        std::vector<float> currentInput = input._input;
        std::vector<float> output;
        for (auto layer : *_layers)
        {
            layer->forwardProp(currentInput, output);
            currentInput = output;
        }
    }

private:
    std::shared_ptr<LayerSet> _layers;
    std::shared_ptr<IDataFeed> _dataFeed;
};

int main()
{   
    // create layers
    std::shared_ptr<LayerSet> layers(new LayerSet({
        std::make_shared<InputLayer>(3),
        std::make_shared<FullyConnectedHiddenLayer>(3, 20),
        std::make_shared<OutputLayer>(20, 2)
    }));

    // create a dummy data set
    std::vector<InputData> staticData = { 
        {{0.5,0.5,0.5}, {0.4,0.4}}, 
        {{0.5,0.5,0.5}, {0.4,0.4}} 
    };

    std::cout << staticData.size() << "   " << staticData[0]._input.size() << std::endl;
    std::shared_ptr<IDataFeed> dataFeed(new StaticDataFeed(staticData));

    auto trainer = std::make_shared<Trainer>(layers, dataFeed);
    trainer->train();
    return 0;
}
