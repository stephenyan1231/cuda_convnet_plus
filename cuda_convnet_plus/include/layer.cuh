/* 
 * Copyright (c) 2011, Alex Krizhevsky (akrizhevsky@gmail.com)
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification,
 * are permitted provided that the following conditions are met:
 *
 * - Redistributions of source code must retain the above copyright notice,
 *   this list of conditions and the following disclaimer.
 * 
 * - Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
 * EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef LAYER_CUH
#define	LAYER_CUH

#include <string>
#include <vector>
#include <map>
#include <assert.h>
#include <nvmatrix.cuh>

#include "convnet.cuh"
#include "cost.cuh"
#include "weights.cuh"
#include "neuron.cuh"

class Cost;
class ConvNet;
class CostLayer;
class DataLayer;

/*
 * Abstract layer.
 */
class Layer {
protected:
	ConvNet* _convNet;
	std::vector<Layer*> _prev, _next;
	int _rcvdFInputs, _rcvdBInputs;
	int _GPU;
#ifdef MULTIGPU
	cudaEvent_t _fpropEvent;
	cudaEvent_t _bpropEvent[MAX_NEXT_LAYER];
	int _bpropEventID;
#endif

	NVMatrixV _inputs;
	NVMatrix *_outputs; // TODO: make this a pointer so you can reuse previous layers' matrices
	NVMatrix *_actsGrad; // Layer activity gradients
	bool _gradConsumer, _foundGradConsumers, _trans;
	bool _conserveMem;
	int _numGradProducersNext;
	int _actsTarget, _actsGradTarget;
	std::string _name, _type;
	void fpropNext(PASS_TYPE passType);
	virtual void truncBwdActs();
	virtual void fpropActs(int inpIdx, float scaleTargets,
			PASS_TYPE passType) = 0;

	virtual void fpropPreCommon(NVMatrixV& v, PASS_TYPE passType);

	virtual void fpropPostCommon(NVMatrixV& v, PASS_TYPE passType);

	virtual void bpropPreCommon(NVMatrix& v, PASS_TYPE passType);

	virtual void bpropActs(NVMatrix& v, int inpIdx, float scaleTargets,
			PASS_TYPE passType) {
		assert(!isGradProducer());
		// Only do nothing if not grad producer
	}

	virtual void bpropPostCommon(NVMatrix& v, PASS_TYPE passType);
public:
	static bool _saveActsGrad, _saveActs;

	Layer(ConvNet* convNet, PyObject* paramsDict, bool trans);

	virtual void fprop(PASS_TYPE passType);
	void fprop(NVMatrix& v, PASS_TYPE passType);
	virtual void fprop(NVMatrixV& v, PASS_TYPE passType);
	virtual void bprop(PASS_TYPE passType);
	void bprop(NVMatrix& v, PASS_TYPE passType);
	virtual void reset();
	int incRcvdBInputs();
	int getRcvdFInputs();
	int getRcvdBInputs();
	bool isGradConsumer();
	virtual bool isGradProducer();
	std::string& getName();
	std::string& getType();
	void addNext(Layer* l);
	void addPrev(Layer* l);
	std::vector<Layer*>& getPrev();
	std::vector<Layer*>& getNext();
	virtual NVMatrix& getActs();
	virtual NVMatrix& getActsGrad();
	virtual void postInit();

	// Do nothing if this layer has no weights
	virtual void updateWeights() {
	}
	virtual void setWeightsEpsScale(float epsW_scale) {

	}
	virtual void multiplyWeightsEpsScale(float multiplier) {

	}

	virtual void checkGradients() {
	}
	virtual void copyToCPU() {
	}
	virtual void copyToGPU() {
#ifdef MULTIGPU
		checkCudaErrors(cudaSetDevice(_GPU));
#endif
	}
#ifdef MULTIGPU
	int getGPU();
	cudaEvent_t getFpropEvent();
	void initBpropEvent();
	cudaEvent_t& getNextBpropEvent();
#endif

};

class NeuronLayer: public Layer {
protected:
	Neuron* _neuron;

	virtual void fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType);
	virtual void bpropActs(NVMatrix& v, int inpIdx, float scaleTargets,
			PASS_TYPE passType);
public:
	NeuronLayer(ConvNet* convNet, PyObject* paramsDict);
};

class WeightLayer: public Layer {
protected:
	WeightList _weights;
	Weights *_biases;
	float _wStep, _bStep;

	void setWeightsEpsScale(float epsScale);
	void multiplyWeightsEpsScale(float multiplier);

	void bpropPreCommon(NVMatrix& v, PASS_TYPE passType);
	virtual void bpropBiases(NVMatrix& v, PASS_TYPE passType) = 0;
	virtual void bpropWeights(NVMatrix& v, int inpIdx, PASS_TYPE passType) = 0;
public:
	WeightLayer(ConvNet* convNet, PyObject* paramsDict, bool trans,
			bool useGrad);
	virtual void updateWeights();
	virtual void copyToCPU();
	virtual void copyToGPU();
	void checkGradients();
	Weights& getWeights(int idx);
};

class ScalingLayer: public WeightLayer {
protected:
	void fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType);
	void bpropActs(NVMatrix& v, int inpIdx, float scaleTargets,
			PASS_TYPE passType);
	void bpropBiases(NVMatrix& v, PASS_TYPE passType);
	void bpropWeights(NVMatrix& v, int inpIdx, PASS_TYPE passType);
public:
	ScalingLayer(ConvNet* convNet, PyObject* paramsDict);
};

class NormalizeLayer: public Layer {
protected:
	NVMatrix _norm;
	void fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType);
	void bpropActs(NVMatrix& v, int inpIdx, float scaleTargets,
			PASS_TYPE passType);
public:
	enum NormalizeType {
		L2, L1
	};
	NormalizeLayer(ConvNet* convNet, PyObject* paramsDict);
};

class FCLayer: public WeightLayer {
protected:
	// dropout binary map
	Matrix _hDoMap;
	NVMatrix _devDoMap;
	float _dropout;

protected:
	void fpropPreCommon(NVMatrixV& v, PASS_TYPE passType);
	void fpropPostCommon(NVMatrixV& v, PASS_TYPE passType);
	void initDropoutMap();
	void bpropPreCommon(NVMatrix& v, PASS_TYPE passType);

	void fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType);
	void bpropActs(NVMatrix& v, int inpIdx, float scaleTargets,
			PASS_TYPE passType);
	void bpropBiases(NVMatrix& v, PASS_TYPE passType);
	void bpropWeights(NVMatrix& v, int inpIdx, PASS_TYPE passType);
public:
	FCLayer(ConvNet* convNet, PyObject* paramsDict);
};

class SoftmaxLayer: public Layer {
protected:
	void fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType);
	void bpropActs(NVMatrix& v, int inpIdx, float scaleTargets,
			PASS_TYPE passType);
public:
	SoftmaxLayer(ConvNet* convNet, PyObject* paramsDict);
};

class LogSoftmaxLayer: public Layer {
protected:
	void fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType);
	void bpropActs(NVMatrix& v, int inpIdx, float scaleTargets,
			PASS_TYPE passType);
public:
	LogSoftmaxLayer(ConvNet* convNet, PyObject* paramsDict);
};

class EltwiseSumLayer: public Layer {
protected:
	vector<float>* _coeffs;
	void fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType);
	void bpropActs(NVMatrix& v, int inpIdx, float scaleTargets,
			PASS_TYPE passType);
public:
	EltwiseSumLayer(ConvNet* convNet, PyObject* paramsDict);
};

class EltwiseMaxLayer: public Layer {
protected:
	void fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType);
	void bpropActs(NVMatrix& v, int inpIdx, float scaleTargets,
			PASS_TYPE passType);
public:
	EltwiseMaxLayer(ConvNet* convNet, PyObject* paramsDict);
};

class DataLayer: public Layer {
private:
	int _dataIdx;
	void fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType);
public:
	DataLayer(ConvNet* convNet, PyObject* paramsDict);

	bool isGradProducer();
	void fprop(PASS_TYPE passType);
	void fprop(NVMatrixV& data, PASS_TYPE passType);
};

class LocalLayer: public WeightLayer {
protected:
	struct FilterConns {
		int* hFilterConns;
		int* dFilterConns;
	};
	vector<FilterConns>* _filterConns;

	intv* _padding, *_stride, *_filterSize, *_channels, *_imgSize, *_groups;
	intv* _imgPixels, *_filterPixels, *_filterChannels, *_overSample,
			*_randSparse;
	int _modulesX, _modules, _numFilters;

	void copyToGPU();

public:
	LocalLayer(ConvNet* convNet, PyObject* paramsDict, bool useGrad);
};

class ConvLayer: public LocalLayer {
protected:
	int _partialSum;
	bool _sharedBiases;

	NVMatrix _weightGradTmp, _actGradTmp;

	void fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType);
	void bpropActs(NVMatrix& v, int inpIdx, float scaleTargets,
			PASS_TYPE passType);
	void bpropBiases(NVMatrix& v, PASS_TYPE passType);
	void bpropWeights(NVMatrix& v, int inpIdx, PASS_TYPE passType);
	void truncBwdActs();
	void fpropPostCommon(NVMatrixV& v, PASS_TYPE passType);

public:
	ConvLayer(ConvNet* convNet, PyObject* paramsDict);
};

class LocalUnsharedLayer: public LocalLayer {
protected:
	NVMatrix _sexMask;
	void fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType);
	void bpropActs(NVMatrix& v, int inpIdx, float scaleTargets,
			PASS_TYPE passType);
	void bpropBiases(NVMatrix& v, PASS_TYPE passType);
	void bpropWeights(NVMatrix& v, int inpIdx, PASS_TYPE passType);
public:
	LocalUnsharedLayer(ConvNet* convNet, PyObject* paramsDict);
};

class PoolLayer: public Layer {
protected:
	int _channels, _sizeX, _start, _stride, _outputsX;
	int _imgSize;
	string _pool;
public:
	PoolLayer(ConvNet* convNet, PyObject* paramsDict, bool trans);

	static PoolLayer& makePoolLayer(ConvNet* convNet, PyObject* paramsDict);
};

class AvgPoolLayer: public PoolLayer {
protected:
	void fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType);
	void bpropActs(NVMatrix& v, int inpIdx, float scaleTargets,
			PASS_TYPE passType);
public:
	AvgPoolLayer(ConvNet* convNet, PyObject* paramsDict);
};

class MaxPoolLayer: public PoolLayer {
protected:
	void fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType);
	void bpropActs(NVMatrix& v, int inpIdx, float scaleTargets,
			PASS_TYPE passType);
public:
	MaxPoolLayer(ConvNet* convNet, PyObject* paramsDict);
};

class NailbedLayer: public Layer {
protected:
	int _channels, _start, _stride, _outputsX;
	int _imgSize;
public:
	void fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType);
	void bpropActs(NVMatrix& v, int inpIdx, float scaleTargets,
			PASS_TYPE passType);

	NailbedLayer(ConvNet* convNet, PyObject* paramsDict);
};

class GaussianBlurLayer: public Layer {
protected:
	int _channels;
	Matrix* _hFilter;
	NVMatrix _filter;
	NVMatrix _actGradsTmp;
public:
	void fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType);
	void bpropActs(NVMatrix& v, int inpIdx, float scaleTargets,
			PASS_TYPE passType);
	void copyToGPU();

	GaussianBlurLayer(ConvNet* convNet, PyObject* paramsDict);
};

class ResizeLayer: public Layer {
protected:
	int _channels;
	float _scale;
	int _imgSize, _tgtSize;
public:
	void fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType);
	void bpropActs(NVMatrix& v, int inpIdx, float scaleTargets,
			PASS_TYPE passType);

	ResizeLayer(ConvNet* convNet, PyObject* paramsDict);
};

class RGBToYUVLayer: public Layer {
public:
	void fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType);
	void bpropActs(NVMatrix& v, int inpIdx, float scaleTargets,
			PASS_TYPE passType);

	RGBToYUVLayer(ConvNet* convNet, PyObject* paramsDict);
};

class RGBToLABLayer: public Layer {
protected:
	bool _center;
public:
	void fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType);
	void bpropActs(NVMatrix& v, int inpIdx, float scaleTargets,
			PASS_TYPE passType);

	RGBToLABLayer(ConvNet* convNet, PyObject* paramsDict);
};

class ResponseNormLayer: public Layer {
protected:
	int _channels, _size;
	float _bias, _scale, _pow;
	NVMatrix _denoms;

	void fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType);
	void bpropActs(NVMatrix& v, int inpIdx, float scaleTargets,
			PASS_TYPE passType);
	void truncBwdActs();
public:
	ResponseNormLayer(ConvNet* convNet, PyObject* paramsDict);
};

class CrossMapResponseNormLayer: public ResponseNormLayer {
protected:
	bool _blocked;
	void fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType);
	void bpropActs(NVMatrix& v, int inpIdx, float scaleTargets,
			PASS_TYPE passType);
public:
	CrossMapResponseNormLayer(ConvNet* convNet, PyObject* paramsDict);
};

class ContrastNormLayer: public ResponseNormLayer {
protected:
	int _imgSize;
	NVMatrix _meanDiffs;

	void fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType);
	void bpropActs(NVMatrix& v, int inpIdx, float scaleTargets,
			PASS_TYPE passType);
	void truncBwdActs();
public:
	ContrastNormLayer(ConvNet* convNet, PyObject* paramsDict);
};

class CostLayer: public Layer {
protected:
	float _coeff;
	doublev _costv;
public:
	CostLayer(ConvNet* convNet, PyObject* paramsDict, bool trans);
	void bprop(PASS_TYPE passType);
	virtual doublev& getCost();
	float getCoeff();
	void setCoeff(float coeff);
	void scaleCoeff(float scale);
	bool isGradProducer();

	static CostLayer& makeCostLayer(ConvNet* convNet, string& type,
			PyObject* paramsDict);
};

/*
 * Input 0: labels
 * Input 1: softmax outputs
 */
class LogregCostLayer: public CostLayer {
protected:
	void fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType);
	void bpropActs(NVMatrix& v, int inpIdx, float scaleTargets,
			PASS_TYPE passType);
public:
	LogregCostLayer(ConvNet* convNet, PyObject* paramsDict);
};

/*
 * Input 0: labels
 * Input 1: softmax outputs
 */
class RegCostLayer: public CostLayer {
protected:
	void fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType);
	void bpropActs(NVMatrix& v, int inpIdx, float scaleTargets,
			PASS_TYPE passType);
public:
	RegCostLayer(ConvNet* convNet, PyObject* paramsDict);
};

/*
 * Input 0: difference
 * */
class SumOfSquaresCostLayer: public CostLayer {
protected:
	void fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType);
	void bpropActs(NVMatrix& v, int inpIdx, float scaleTargets,
			PASS_TYPE passType);
public:
	SumOfSquaresCostLayer(ConvNet* convNet, PyObject* paramsDict);
};

/* cost is sum of squares of differences
 * Input 0: labels
 * Input 1: predicted continuous-valued labels
 */
class SumOfSquaresOfDiffCostLayer: public CostLayer {
protected:
	bool _relativeDiff;
	void fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType);
	void bpropActs(NVMatrix& v, int inpIdx, float scaleTargets,
			PASS_TYPE passType);
public:
	SumOfSquaresOfDiffCostLayer(ConvNet* convNet, PyObject* paramsDict);
};

// cost is shifted logistic functino of sum of squares of differences. f(x)=1/(1+e^(-ax))-0.5  x=0.5 * \sum_i ||p_i-q_i||^2
class LogSumOfSquaresOfDiffCostLayer: public CostLayer {
protected:
	float _scale;
	void fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType);
	void bpropActs(NVMatrix& v, int inpIdx, float scaleTargets,
			PASS_TYPE passType);
public:
	LogSumOfSquaresOfDiffCostLayer(ConvNet* convNet, PyObject* paramsDict);
};

class CroppingLayer: public Layer {
protected:
	int _channels, _start, _end;

	void fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType);
	void bpropActs(NVMatrix& v, int inpIdx, float scaleTargets,
			PASS_TYPE passType);
public:
	CroppingLayer(ConvNet* convNet, PyObject* paramsDict);
};

class ColorEnhanceCostLayer: public CostLayer {
protected:
	void fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType);
	void bpropActs(NVMatrix& v, int inpIdx, float scaleTargets,
			PASS_TYPE passType);
public:
	ColorEnhanceCostLayer(ConvNet* convNet, PyObject* paramsDict);
};

// measure Lab space L2 distance. L,a,b channel transform are regressed separately
class ColorEnhanceSeparateCostLayer: public CostLayer {
protected:
	void fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType);
	void bpropActs(NVMatrix& v, int inpIdx, float scaleTargets,
			PASS_TYPE passType);
public:
	ColorEnhanceSeparateCostLayer(ConvNet* convNet, PyObject* paramsDict);
};

class ConcatenateLayer: public Layer {
protected:
	int _outputs_p;
	void fpropPreCommon(NVMatrixV& v, PASS_TYPE passType);
	void fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType);
	void bpropPreCommon(NVMatrix& v, PASS_TYPE passType);
	void bpropActs(NVMatrix& v, int inpIdx, float scaleTargets,
			PASS_TYPE passType);

public:
	ConcatenateLayer(ConvNet* convNet, PyObject* paramsDict);
};

class GradMagEnhanceCostLayer: public CostLayer{
protected:
	NVMatrix logInL;
	NVMatrix predEnhGradMag;
	void fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType);
	void bpropActs(NVMatrix& v, int inpIdx, float scaleTargets,
			PASS_TYPE passType);
public:
	GradMagEnhanceCostLayer(ConvNet* convNet, PyObject* paramsDict);
};

#endif	/* LAYER_CUH */

