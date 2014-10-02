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
#include <cutil_inline.h>
#include <iostream>

#include <layer_kernels.cuh>
#include <layer.cuh>
#include <data.cuh>
#include <util.cuh>
#include <cudaconv2.cuh>
#include <matrix.h>
#include <GPUmonitor.h>

#include <cmath>
#include <algorithm>
#include <vector>
#include <cstring>

using namespace std;
extern ConvNet* model;
extern GPUmonitor *gpuMonitor;

float thres = 1.0e5;
bool verbose = 0;

#define checkNVMatrixNan(dmat,msg) _checkNVMatrixNan(dmat,msg,__FILE__,__LINE__)

void _checkNVMatrixNan(NVMatrix &dMat, string msg, const char* filenm,
		const int linenum) {
	//int leadingDim	= dMat.getLeadingDim();
	//int stride = dMat.getStride();
	//int followingDim = dMat.getFollowingDim();
	//printf("ldDim:%d stride:%d followingDim:%d\n",leadingDim,stride,followingDim);
	if (isnan(dMat.sum())) {
		printf("_checkNVMatrixNan File:%s line:%d\n", filenm, linenum);
		dMat.printShape(msg.c_str());
		dMat.fprint(msg.c_str(), dMat.getNumRows(), dMat.getNumCols());
		dMat.print(2, 2);
		printf("min:%f max:%f mean:%f\n", dMat.min(), dMat.max(), dMat.mean());
		exit(1);
	}

}

/* 
 * =======================
 * Layer
 * =======================
 */

Layer::Layer(ConvNet* convNet, PyObject* paramsDict, bool trans) :
		_convNet(convNet), _trans(trans) {
	_name = pyDictGetString(paramsDict, "name");
	_type = pyDictGetString(paramsDict, "type");

	_numGradProducersNext = 0;
	_foundGradConsumers = false;
	_gradConsumer = pyDictGetInt(paramsDict, "gradConsumer");
	_actsTarget = pyDictGetInt(paramsDict, "actsTarget");
	_actsGradTarget = pyDictGetInt(paramsDict, "actsGradTarget");
	_conserveMem = pyDictGetInt(paramsDict, "conserveMem");
	_outputs = _actsTarget < 0 ? new NVMatrix() : NULL;
	_actsGrad = _actsGradTarget < 0 ? new NVMatrix() : NULL; // can be commented out since it's done in postInit()

	_GPU = pyDictGetInt(paramsDict, "GPU");

#ifdef MULTIGPU
	cudaSetDevice(_GPU);
	NVMatrix::checkCUDAError(cudaEventCreate(&_fpropEvent),
			"Layer::Layer Layer::Layer");
#endif
	printf(" init layer: %s\n", _name.c_str());
	printf("_actsTarget:%d _actsGradTarget:%d _GPU:%d\n", _actsTarget,
			_actsGradTarget, _GPU);
}

void Layer::fpropNext(PASS_TYPE passType) {
	for (int i = 0; i < _next.size(); i++) {
//		printf("_next[%d] name:%s \n",i,_name.c_str());
//		if(&_next[i]->getActs())
//			printf("_next[%d] name:%s rows:%d cols:%d\n",i,_name.c_str(),
//					_next[i]->getActs().getNumRows(),_next[i]->getActs().getNumCols());
		_next[i]->fprop(passType);
	}
}

void Layer::truncBwdActs() {
	// Only truncate actsGrad if I own it
	if (_conserveMem && _actsGradTarget < 0) {
		getActsGrad().truncate();
	}
	if (_conserveMem) {
		getActs().truncate();
	}
}

void Layer::fpropPreCommon(NVMatrixV& v, PASS_TYPE passType) {
	// Do nothing by default
}

void Layer::fpropPostCommon(NVMatrixV& v, PASS_TYPE passType) {

}

void Layer::bpropPreCommon(NVMatrix& v, PASS_TYPE passType) {
#ifdef MULTIGPU
	NVMatrix::checkCUDAError(cudaSetDevice(_GPU),
			"Layer::bpropPreCommon cudaSetDevice");
#endif
	// Do nothing by default

}

void Layer::bpropPostCommon(NVMatrix& v, PASS_TYPE passType) {
#ifdef MULTIGPU
	NVMatrix::checkCUDAError(cudaSetDevice(_GPU),
			"Layer::bpropPostCommon cudaSetDevice");
#endif
	//float gradMin=getActsGrad().min();
	//float gradMax=getActsGrad().max();
	//float gradMean=getActsGrad().mean();
	//if(abs(gradMin)>thres || abs(gradMax)>thres){
	//	printf("layer:%s gradMin:%f gradMax:%f gradMean:%f\n",_name.c_str(),
	//		gradMin,gradMax,gradMean);
	//}
}

void Layer::fprop(PASS_TYPE passType) {
	_rcvdFInputs += 1;
	if (_rcvdFInputs == _prev.size()) {
#ifdef MULTIGPU
		//reset
		_bpropEventID = 0;
		NVMatrix::checkCUDAError(cudaSetDevice(_GPU),
				"Layer::fprop(PASS_TYPE passType) cudaSetDevice");
		for (int i = 0; i < _prev.size(); ++i) {
			NVMatrix::checkCUDAError(
					cudaEventSynchronize(_prev[i]->getFpropEvent()),
					"Layer::fprop(PASS_TYPE passType) (cudaEventSynchronize");
		}
#endif
		NVMatrixV v;
		for (int i = 0; i < _prev.size(); i++) {
			v.push_back(&_prev[i]->getActs());
		}
		fprop(v, passType);
	}
}

void Layer::fprop(NVMatrix& v, PASS_TYPE passType) {
	NVMatrixV vl;
	vl.push_back(&v);
	fprop(vl, passType);
}

void Layer::fprop(NVMatrixV& v, PASS_TYPE passType) {
//	printf("Layer %s\n",_name.c_str());
//	if(_actsTarget<0)
//		printf("Layer %s fprop. acts rows:%d cols:%d\n",_name.c_str(),
//					getActs().getNumRows(),getActs().getNumCols());

	assert(v.size() == _prev.size());
	_inputs.clear();
	_inputs.insert(_inputs.begin(), v.begin(), v.end());
	_outputs = _actsTarget < 0 ? _outputs : _inputs[_actsTarget];
	_rcvdFInputs = _prev.size();
	for (NVMatrixV::iterator it = v.begin(); it != v.end(); ++it) {
		(*it)->transpose(_trans);
	}
	getActs().transpose(_trans);

//	printf("Layer %s fprop. acts rows:%d cols:%d\n",_name.c_str(),
//			getActs().getNumRows(),getActs().getNumCols());

	fpropPreCommon(v, passType);

	// First do fprop on the input whose acts matrix I'm sharing, if any
	if (_actsTarget >= 0) {
		fpropActs(_actsTarget, 0, passType);
	}
	// Then add the rest of the inputs to that
	for (int i = 0; i < _prev.size(); i++) {
		if (i != _actsTarget) {
			fpropActs(i, _actsTarget >= 0 || i > 0, passType);
		}
	}

	if (verbose) {
		NVMatrix tmp(getActs());
		getActs().apply(NVMatrixOps::Abs(), tmp);
		float mean_abs_act = tmp.sum() / tmp.getNumElements();
		printf("Layer::fprop %s. mean_abs_act:%f\n", _name.c_str(),
				mean_abs_act);
	}

	//float actMax=getActs().max(),actMin=getActs().min();
	//if(abs(actMax)>thres || abs(actMin)>thres){
	//	printf("\nlayer:%s actMax:%f actMin:%f\n",_name.c_str(),actMax,actMin);
	//	for(int i=0;i<_inputs.size();++i){
	//		float inputMax=(*_inputs[i]).max();
	//		float inputMin=(*_inputs[i]).min();
	//		printf("input:%d inputMax:%f inputMin:%f\n",i,inputMax,inputMin);
	//	}
	//}

	fpropPostCommon(v, passType);

#ifdef MULTIGPU
	NVMatrix::checkCUDAError(cudaEventRecord(_fpropEvent),
			"Layer::fprop(NVMatrixV& v, PASS_TYPE passType) cudaEventRecord");
#endif
	fpropNext(passType);
}

void Layer::bprop(PASS_TYPE passType) {
	if (_rcvdBInputs == _numGradProducersNext) {
		_rcvdBInputs++; // avoid doing bprop computation twice
#ifdef MULTIGPU
		for (int i = 0; i < _next.size(); ++i) {
			NVMatrix::checkCUDAError(cudaEventSynchronize(_bpropEvent[i]),
					"Layer::bprop(PASS_TYPE passType) cudaEventSynchronize (_bpropEvent[i])");
		}
#endif
		bprop(getActsGrad(), passType);
	}
}

void Layer::bprop(NVMatrix& v, PASS_TYPE passType) {
	if (verbose & v.getNumElements() > 0) {
		NVMatrix tmp(v);
		v.apply(NVMatrixOps::Abs(), tmp);
		float meanAbs = tmp.sum() / tmp.getNumElements();
		printf("Layer::bprop %s v rows,cols,%d %d mean Abs:%f\n", _name.c_str(),
				v.getNumRows(), v.getNumCols(), meanAbs);
	}

	v.transpose(_trans);
	for (int i = 0; i < _prev.size(); i++) {
		_prev[i]->getActs().transpose(_trans);
		_prev[i]->getActsGrad().transpose(_trans);
	}
	getActs().transpose(_trans);

	bpropPreCommon(v, passType);

	if (isGradProducer()) {
		// First propagate activity gradient to all layers whose activity
		// gradient matrix I'm definitely not sharing.
		for (int i = 0; i < _prev.size(); i++) {
			if (_prev[i]->isGradConsumer() && _actsGradTarget != i) {
#ifdef MULTIGPU
				// do bprop on previous layer's device.
				// in the case where previous layer connects to multiple subsequent layers,
				// there is no need to synchronize the bprop for a previous layer
				NVMatrix::checkCUDAError(cudaSetDevice(_prev[i]->getGPU()),
						"Layer::bprop(NVMatrix& v, PASS_TYPE passType) cudaSetDevice(_prev[i]->getGPU())");
#endif
				bpropActs(v, i, _prev[i]->getRcvdBInputs() > 0 ? 1 : 0,
						passType);
				_prev[i]->incRcvdBInputs();
#ifdef MULTIGPU
				NVMatrix::checkCUDAError(
						cudaEventRecord(_prev[i]->getNextBpropEvent()),
						"Layer::bprop(NVMatrix& v, PASS_TYPE passType) cudaEventRecord");
#endif
			}
		}

		// Then propagate activity gradient to the layer whose activity gradient
		// matrix I'm sharing, if any.
		if (_actsGradTarget >= 0 && _prev[_actsGradTarget]->isGradConsumer()) {
#ifdef MULTIGPU
			NVMatrix::checkCUDAError(
					cudaSetDevice(_prev[_actsGradTarget]->getGPU()),
					"Layer::bprop(NVMatrix& v, PASS_TYPE passType) cudaSetDevice(_prev[_actsGradTarget]->getGPU())");
#endif
			bpropActs(v, _actsGradTarget,
					_prev[_actsGradTarget]->getRcvdBInputs() > 0 ? 1 : 0,
					passType);
			_prev[_actsGradTarget]->incRcvdBInputs();
#ifdef MULTIGPU
			NVMatrix::checkCUDAError(
					cudaEventRecord(
							_prev[_actsGradTarget]->getNextBpropEvent()),
					"Layer::bprop(NVMatrix& v, PASS_TYPE passType) cudaEventRecord(_prev[_actsGradTarget]->getNextBpropEvent())");
#endif
		}
	}
	truncBwdActs();

	bpropPostCommon(v, passType);

	if (isGradProducer()) {
		for (int i = 0; i < _prev.size(); i++) {
			if (_prev[i]->isGradConsumer()) {
				_prev[i]->bprop(passType);
			}
		}
	}
}

void Layer::reset() {
	_rcvdFInputs = 0;
	_rcvdBInputs = 0;
}

string& Layer::getName() {
	return _name;
}

string& Layer::getType() {
	return _type;
}

int Layer::getRcvdFInputs() {
	return _rcvdFInputs;
}

int Layer::getRcvdBInputs() {
	return _rcvdBInputs;
}

int Layer::incRcvdBInputs() {
	return ++_rcvdBInputs;
}

void Layer::addNext(Layer* l) {
	_next.push_back(l);
	_numGradProducersNext += l->isGradProducer();
}

void Layer::addPrev(Layer* l) {
	_prev.push_back(l);
}

void Layer::postInit() {
//    _outputs = _actsTarget < 0 ? new NVMatrix() : &_prev[_actsTarget]->getActs();
	_actsGrad =
			_actsGradTarget < 0 ?
					new NVMatrix() : &_prev[_actsGradTarget]->getActsGrad();
}

#ifdef MULTIGPU
int Layer::getGPU() {
	return _GPU;
}

cudaEvent_t Layer::getFpropEvent() {
	return _fpropEvent;
}

void Layer::initBpropEvent() {
	NVMatrix::checkCUDAError(cudaSetDevice(_GPU),
			"Layer::initBpropEvent cudaSetDevice");
	for (int i = 0; i < _next.size(); ++i) {
		NVMatrix::checkCUDAError(cudaEventCreate(&_bpropEvent[i]),
				"Layer::initBpropEvent() cudaEventCreate(&_bpropEvent[i])");
	}
}
cudaEvent_t& Layer::getNextBpropEvent() {
	return _bpropEvent[_bpropEventID++];
}

#endif
// Does this layer, or some layer below it, need the gradient
// for parameter updates?
// Only weight layers should be grad consumers themselves.
bool Layer::isGradConsumer() {
	if (!_foundGradConsumers) {
		for (int i = 0; i < _prev.size(); i++) {
			_gradConsumer |= _prev[i]->isGradConsumer();
		}
		_foundGradConsumers = true;
	}
	return _gradConsumer;
}

// Does this layer produce gradient for layers below?
bool Layer::isGradProducer() {
	return true;
}

vector<Layer*>& Layer::getPrev() {
	return _prev;
}

vector<Layer*>& Layer::getNext() {
	return _next;
}

NVMatrix& Layer::getActs() {
	assert(_outputs != NULL);
	return *_outputs;
}

NVMatrix& Layer::getActsGrad() {
	assert(_actsGrad != NULL);
	return *_actsGrad;
}

/* 
 * =======================
 * NeuronLayer
 * =======================
 */
NeuronLayer::NeuronLayer(ConvNet* convNet, PyObject* paramsDict) :
		Layer(convNet, paramsDict, true) {
	_neuron = &Neuron::makeNeuron(PyDict_GetItemString(paramsDict, "neuron"));
}

void NeuronLayer::bpropActs(NVMatrix& v, int inpIdx, float scaleTargets,
		PASS_TYPE passType) {
	_neuron->computeInputGrad(v, _prev[0]->getActsGrad(), scaleTargets > 0);
}

void NeuronLayer::fpropActs(int inpIdx, float scaleTargets,
		PASS_TYPE passType) {
	_neuron->activate(*_inputs[0], getActs());
}

/* 
 * =======================
 * WeightLayer
 * =======================
 */
WeightLayer::WeightLayer(ConvNet* convNet, PyObject* paramsDict, bool trans,
		bool useGrad) :
		Layer(convNet, paramsDict, trans) {

	MatrixV& hWeights = *pyDictGetMatrixV(paramsDict, "weights");
	MatrixV& hWeightsInc = *pyDictGetMatrixV(paramsDict, "weightsInc");
	Matrix& hBiases = *pyDictGetMatrix(paramsDict, "biases");
	Matrix& hBiasesInc = *pyDictGetMatrix(paramsDict, "biasesInc");

	floatv& momW = *pyDictGetFloatV(paramsDict, "momW");
	float momB = pyDictGetFloat(paramsDict, "momB");
	floatv& epsW = *pyDictGetFloatV(paramsDict, "epsW");
	float epsB = pyDictGetFloat(paramsDict, "epsB");
	floatv& wc = *pyDictGetFloatV(paramsDict, "wc");
	float wcBias = pyDictGetFloat(paramsDict, "wcB");

	printf("(epsW,momW,wc) for %d inputs\n", epsW.size());
	for (int i = 0; i < momW.size(); ++i) {
		printf("(%.12f,%.12f,%.12f) ", epsW[i], momW[i], wc[i]);
	}
	printf("\n");
	printf("momB:%.12f epsB:%.12f wcBias:%.12f\n", momB, epsB, wcBias);

	// Source layers for shared weights
	intv& weightSourceLayerIndices = *pyDictGetIntV(paramsDict,
			"weightSourceLayerIndices");
	// Weight matrix indices (inside the above source layers) for shared weights
	intv& weightSourceMatrixIndices = *pyDictGetIntV(paramsDict,
			"weightSourceMatrixIndices");

	for (int i = 0; i < weightSourceLayerIndices.size(); i++) {
		int srcLayerIdx = weightSourceLayerIndices[i];
		int matrixIdx = weightSourceMatrixIndices[i];
		if (srcLayerIdx == convNet->getNumLayers()) { // Current layer
			_weights.addWeights(*new Weights(_weights[matrixIdx], epsW[i]));
		} else if (srcLayerIdx >= 0) {
			WeightLayer& srcLayer =
					*static_cast<WeightLayer*>(&convNet->getLayer(srcLayerIdx));
			Weights* srcWeights = &srcLayer.getWeights(matrixIdx);
			_weights.addWeights(*new Weights(*srcWeights, epsW[i]));
		} else {
			_weights.addWeights(
					*new Weights(*hWeights[i], *hWeightsInc[i], epsW[i], wc[i],
							momW[i], useGrad));
		}
	}

	_biases = new Weights(hBiases, hBiasesInc, epsB, wcBias, momB, true);

	// Epsilons for finite-difference gradient checking operation
	_wStep = 0.001;
	_bStep = 0.002;

	delete &weightSourceLayerIndices;
	delete &weightSourceMatrixIndices;
	delete &hWeights;
	delete &hWeightsInc;
	delete &momW;
	delete &epsW;
	delete &wc;
}

void WeightLayer::setWeightsEpsScale(float eps_scale) {
	Layer::setWeightsEpsScale(eps_scale);
	_weights.setEpsScale(eps_scale);
	_biases->setEpsScale(eps_scale);
}

void WeightLayer::multiplyWeightsEpsScale(float multiplier) {
	Layer::multiplyWeightsEpsScale(multiplier);
	_weights.multiplyEpsScale(multiplier);
	_biases->multiplyEpsScale(multiplier);
}

void WeightLayer::bpropPreCommon(NVMatrix& v, PASS_TYPE passType) {
	Layer::bpropPreCommon(v, passType);

//	printf("WeightLayer::bpropPreCommo %s\n",_name.c_str());

	if (_biases->getEps() > 0) {
		bpropBiases(v, passType);
	}
	for (int i = 0; i < _weights.getSize(); i++) {
		if (_weights[i].getEps() > 0) {
			bpropWeights(v, i, passType);
			// Increment its number of updates
			_weights[i].incNumUpdates();
		}
	}
}

void WeightLayer::updateWeights() {
	Layer::updateWeights();
	if (verbose)
		printf("WeightLayer::updateWeights %s\n", _name.c_str());
	_weights.update();
	_biases->update();
}

void WeightLayer::copyToCPU() {
	Layer::copyToCPU();
	_weights.copyToCPU();
	_biases->copyToCPU();
}

void WeightLayer::copyToGPU() {
	Layer::copyToGPU();
	_weights.copyToGPU();
	_biases->copyToGPU();
}

void WeightLayer::checkGradients() {
	for (int i = 0; i < _weights.getSize(); i++) {
		_convNet->checkGradient(_name + " weights[" + tostr(i) + "]", _wStep,
				_weights[i]);
	}
	_convNet->checkGradient(_name + " biases", _bStep, *_biases);
}

Weights& WeightLayer::getWeights(int idx) {
	return _weights[idx];
}

/*
 * ============
 * ScalingLayer Layer
 * ============
 * */

ScalingLayer::ScalingLayer(ConvNet* convNet, PyObject* paramsDict) :
		WeightLayer(convNet, paramsDict, false, true) {

}

void ScalingLayer::fpropActs(int inpIdx, float scaleTargets,
		PASS_TYPE passType) {
	Weights& scaleWeight = _weights[inpIdx];
	scaleWeight.copyToCPU();
	Matrix& hScale = scaleWeight.getCPUW();
	float *scale = hScale.getData();

	getActs().add(*_inputs[inpIdx], scaleTargets, *scale);
	if (scaleTargets == 0) {
		_biases->copyToCPU();
		Matrix& hBias = _biases->getCPUW();
		float *bias = hBias.getData();
		printf("ScalingLayer %s. scale: %f bias:%f scaleTargets:%f\n",
				_name.c_str(), *scale, *bias, scaleTargets);
		getActs().addScalar(*bias);
	}

	if (verbose) {
		NVMatrix tmp(getActs());
		getActs().apply(NVMatrixOps::Abs(), tmp);
		float mean_abs_act = tmp.sum() / tmp.getNumElements();
		printf("ScalingLayer %s. mean_abs_act:%f\n", _name.c_str(),
				mean_abs_act);

	}
}

void ScalingLayer::bpropActs(NVMatrix& v, int inpIdx, float scaleTargets,
		PASS_TYPE passType) {
	Weights& scaleWeight = _weights[inpIdx];
	scaleWeight.copyToCPU();
	Matrix& hScale = scaleWeight.getCPUW();
	float *scale = hScale.getData();
//	printf("ScalingLayer::bpropAct scale:%f\n",*scale);
//	_prev[inpIdx]->getActsGrad().resize(v);
	_prev[inpIdx]->getActsGrad().add(v, scaleTargets, *scale);
}
void ScalingLayer::bpropBiases(NVMatrix& v, PASS_TYPE passType) {
	int numCases = v.getNumCols();
	float scaleBGrad = passType == PASS_GC ? 1 : _biases->getEps() / numCases;

//	printf("ScalingLayer::bpropBiases _biases->getGrad() rows:%d cols:%d\n",
//			_biases->getGrad().getNumRows(),_biases->getGrad().getNumCols());
	_biases->getGrad().resize(1, 1);
	_biases->getGrad().scale(0);
	_biases->getGrad().addScalar(scaleBGrad * v.sum());
}

void ScalingLayer::bpropWeights(NVMatrix& v, int inpIdx, PASS_TYPE passType) {
	int numCases = v.getNumCols();
	NVMatrix a;
//	printf("ScalingLayer::bpropWeights v rows:%d cols:%d getActs() rows:%d cols:%d\n",
//			v.getNumRows(),v.getNumCols(),getActs().getNumRows(),getActs().getNumCols());
	_prev[inpIdx]->getActs().eltwiseMult(v, a);
	float scaleGrad =
			passType == PASS_GC ? 1 : _weights[inpIdx].getEps() / numCases;
//	printf("ScalingLayer::bpropWeights _weights[inpIdx].getGrad() rows:%d cols:%d \n",
//			_weights[inpIdx].getGrad().getNumRows(),_weights[inpIdx].getGrad().getNumCols());
	_weights[inpIdx].getGrad().resize(1, 1);
	_weights[inpIdx].getGrad().setZero();

	//Matrix tmp;
	//_weights[inpIdx].getGrad().copyToHost(tmp,true);
	//float *data= tmp.getData();
	//printf("tmp rows,cols:%d %d data [0]:%f\n",tmp.getNumRows(),tmp.getNumCols(),data[0]);

	printf(
			"ScalingLayer::bpropWeights %s. _weights[inpIdx].getGrad().sum():%f scaleGrad:%f weight grad:%f\n",
			_name.c_str(), _weights[inpIdx].getGrad().sum(), scaleGrad,
			scaleGrad * a.sum());
	_weights[inpIdx].getGrad().addScalar(scaleGrad * a.sum());
}

/*
 * ============
 * NormalizeLayer
 * ============
 */
NormalizeLayer::NormalizeLayer(ConvNet* convNet, PyObject* paramsDict) :
		Layer(convNet, paramsDict, false) {

}

void NormalizeLayer::fpropActs(int inpIdx, float scaleTargets,
		PASS_TYPE passType) {
	assert(inpIdx == 0);
	NVMatrix squared;
	_inputs[inpIdx]->apply(NVMatrixOps::Square(), squared);
	squared.sum(0, _norm);
	_norm.apply(NVMatrixOps::Sqrt());
	if (verbose) {
		printf(
				"NormalizeLayer::fpropActs squared size:%d %d norm size:%d %d norm mean:%f\n",
				squared.getNumRows(), squared.getNumCols(), _norm.getNumRows(),
				_norm.getNumCols(), _norm.sum() / _norm.getNumElements());
		//_norm.print(_norm.getNumRows(), _norm.getNumCols());
	}
	if (_norm.min() == 0) {
		printf("zero norm\n");
		exit(1);
	}
	_inputs[inpIdx]->eltwiseDivideByVector(_norm, getActs());
}

void NormalizeLayer::bpropActs(NVMatrix& v, int inpIdx, float scaleTargets,
		PASS_TYPE passType) {
	assert(inpIdx == 0);
	computeL2NormalizeGrad(_prev[inpIdx]->getActs(), getActs(), _norm, v,
			_prev[inpIdx]->getActsGrad(), scaleTargets == 1);
//
//	NVMatrix prev_v;
//	v.eltwiseMultByVector(_norm,prev_v);
//	_prev[inpIdx]->getActsGrad().add(prev_v,scaleTargets,1);
}

/* 
 * =======================
 * FCLayer
 * =======================
 */
FCLayer::FCLayer(ConvNet* convNet, PyObject* paramsDict) :
		WeightLayer(convNet, paramsDict, true, false) {
	//printf("set dropout to false\n");
	//_dropout=0;
	_dropout = pyDictGetFloat(paramsDict, "dropout");
	_wStep = 0.1;
	_bStep = 0.01;
}

void FCLayer::fpropPreCommon(NVMatrixV& v, PASS_TYPE passType) {
	if (passType == PASS_TRAIN && _dropout > 0)
		//printf("FCLayer::fpropPreCommon dropout in training is enabled\n");
		initDropoutMap();
}

void FCLayer::fpropPostCommon(NVMatrixV& v, PASS_TYPE passType) {
	Layer::fpropPostCommon(v, passType);

	//float actMax=(getActs().max());
	//float actMin=(getActs().min());
	//if(abs(actMax)>thres || abs(actMin)>thres){

	//	float wMin,wMax,wMean;
	//	wMin=(*_weights[0]).min();
	//	wMax=(*_weights[0]).max();
	//	wMean=(*_weights[0]).mean();
	//	printf("layer:%s wMin,wMax,wMean:%f %f %f\n",
	//		_name.c_str(),wMin,wMax,wMean);
	//}

	if (passType == PASS_TRAIN && _dropout > 0) {
		//printf("FCLayer::fpropPostCommon dropout in training is enabled\n");
		getActs().eltwiseMultByVector(_devDoMap);
		//printf("dropout binary map\n");
		//_hDoMap.print(_hDoMap.getNumRows(),_hDoMap.getNumCols());
		//printf("FCLayer::fpropPostCommon activation after dropout\n");
		//getActs().print(getActs().getNumRows(),getActs().getNumCols());
	} else if (passType == PASS_TEST && _dropout > 0) {
		//printf("FCLayer::fpropPostCommon dropout in testing is enabled\n");
		getActs().scale(1 - _dropout);
	}
}

void FCLayer::initDropoutMap() {
	assert(_weights.getSize() > 0);
	int numOut = _weights[0].getNumCols();
	//int numIn=_weights[0].getNumRows();
	//printf("initDropoutMap numIN:%d numOut:%d\n",numIn,numOut);
	_hDoMap.resize(1, numOut);
	vector<int> nums(numOut);
	for (int i = 0; i < numOut; ++i)
		nums[i] = i;
	std::random_shuffle(nums.begin(), nums.end());
	float *hDoMapData = _hDoMap.getData();
	memset(hDoMapData, 0, sizeof(float) * numOut);
	for (int i = 0; i < floor(numOut * (1 - _dropout)); ++i)
		hDoMapData[nums[i]] = 1.0f;
	_devDoMap.resize(_hDoMap);
	_devDoMap.copyFromHost(_hDoMap);

}

void FCLayer::bpropPreCommon(NVMatrix& v, PASS_TYPE passType) {
	WeightLayer::bpropPreCommon(v, passType);
	if (passType == PASS_TRAIN && _dropout > 0) {
		assert(_weights.getSize() > 0);
		//int numOut=_weights[0].getNumCols();
		//assert(numOut==v.getNumCols());
		v.eltwiseMultByVector(_devDoMap);
		//v.print(v.getNumRows(),v.getNumCols());
	}
}

void FCLayer::fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType) {
	checkNVMatrixNan(*_inputs[inpIdx], _name);
	checkNVMatrixNan(*_weights[inpIdx], _name);

	//if(_name==string("fc4096_1"))
	//{
	//	printf("(*_inputs[inpIdx]) min,max,mean:%f %f %f\n",(*_inputs[inpIdx]).min(),(*_inputs[inpIdx]).max(),(*_inputs[inpIdx]).mean());
	//	printf("(*_weights[inpIdx]) min,max,mean:%f %f %f\n",(*_weights[inpIdx]).min(),(*_weights[inpIdx]).max(),(*_weights[inpIdx]).mean());
	//	/*
	//	(*_inputs[inpIdx]).fprint((_name+string("FCLayer fpropActs  _inputs[inpIdx]")).c_str(),(*_inputs[inpIdx]).getNumRows(),
	//	(*_inputs[inpIdx]).getNumCols());
	//(*_weights[inpIdx]).fprint((_name+string("FCLayer fpropActs   _weights[inpIdx]")).c_str(),(*_weights[inpIdx]).getNumRows(),
	//	(*_weights[inpIdx]).getNumCols());*/
	//}

	getActs().addProduct(*_inputs[inpIdx], *_weights[inpIdx], scaleTargets, 1);
	if (scaleTargets == 0) {
		getActs().addVector(_biases->getW());
	}
//	if(_name == string("reglayer")){
//		printf("FCLayer %s. print getActs()\n", _name.c_str());
//		getActs().print(1,getActs().getNumCols());
//	}

	if (verbose) {
		NVMatrix tmp(getActs());
		getActs().apply(NVMatrixOps::Abs(), tmp);
		float mean_abs_act = tmp.sum() / tmp.getNumElements();
		printf("FCLayer %s. mean_abs_act:%f\n", _name.c_str(), mean_abs_act);
	}

//	if (isnan(getActs().sum())) {
//		printf("FCLayer::fpropActs isnan(getActs().sum())\n");
//		printf("inputs min:%f max:%f mean:%f\n", (*_inputs[inpIdx]).min(),
//				(*_inputs[inpIdx]).max(), (*_inputs[inpIdx]).mean());
//		printf("(*_weights[inpIdx]) min:%f max:%f mean:%f\n",
//				(*_weights[inpIdx]).min(), (*_weights[inpIdx]).max(),
//				(*_weights[inpIdx]).mean());
//		printf("_biases min:%f max:%f mean:%f\n", (_biases->getW()).min(),
//				(_biases->getW()).max(), (_biases->getW()).mean());
//		(*_inputs[inpIdx]).fprint(
//				(_name + string("FCLayer fpropActs  _inputs")).c_str(),
//				(*_inputs[inpIdx]).getNumRows(),
//				(*_inputs[inpIdx]).getNumCols());
//		(*_weights[inpIdx]).fprint(
//				(_name + string("FCLayer fpropActs   _weights")).c_str(),
//				(*_weights[inpIdx]).getNumRows(),
//				(*_weights[inpIdx]).getNumCols());
//		(_biases->getW()).fprint(
//				(_name + string("FCLayer fpropActs  _biases")).c_str(),
//				(_biases->getW()).getNumRows(), (_biases->getW()).getNumCols());
//
//	}
//	checkNVMatrixNan(getActs(), _name);
}

void FCLayer::bpropActs(NVMatrix& v, int inpIdx, float scaleTargets,
		PASS_TYPE passType) {
	NVMatrix& weights_T = _weights[inpIdx].getW().getTranspose();
	_prev[inpIdx]->getActsGrad().addProduct(v, weights_T, scaleTargets, 1);
	//checkNVMatrixNan(_prev[inpIdx]->getActsGrad(),_name);
	delete &weights_T;
}

void FCLayer::bpropBiases(NVMatrix& v, PASS_TYPE passType) {
	//if(verbose)
	//	printf("FCLayer::bpropBiases v rows(numCases) cols:%d %d\n",
	//			v.getNumRows(),v.getNumCols());

	int numCases = v.getNumRows();
	float scaleBGrad = passType == PASS_GC ? 1 : _biases->getEps() / numCases;
	_biases->getGrad().addSum(v, 0, 0, scaleBGrad);

	if (0) {
		Matrix biasGrad;
		_biases->getGrad().copyToHost(biasGrad, true);
		float *biasGradData = biasGrad.getData();
		int posNum = 0, negNum = 0, zeroNum = 0;
		for (int i = 0; i < biasGrad.getNumElements(); ++i) {
			if (biasGradData[i] < 0)
				negNum++;
			else if (biasGradData[i] == 0)
				zeroNum++;
			else
				posNum++;
		}
		float biasGradSum = biasGrad.sum();
		printf("FCLayer: %s biasGradSum:%.8f negNum:%d zeroNum:%d posNum:%d\n",
				_name.c_str(), biasGradSum, negNum, zeroNum, posNum);

		Matrix bias;
		_biases->getW().copyToHost(bias, true);
		float *biasData = bias.getData();
		posNum = 0;
		negNum = 0;
		zeroNum = 0;
		for (int i = 0; i < bias.getNumElements(); ++i) {
			if (biasData[i] < 0)
				negNum++;
			else if (biasData[i] == 0)
				zeroNum++;
			else
				posNum++;
		}
		float biasSum = bias.sum();
		printf("FCLayer: %s biasSum:%.8f negNum:%d zeroNum:%d posNum:%d\n",
				_name.c_str(), biasSum, negNum, zeroNum, posNum);

		NVMatrix dbgMat(_biases->getGrad(), true);
		dbgMat.eltwiseDivide(_biases->getW());
		dbgMat.apply(NVMatrixOps::Abs());
		printf("FCLayer: %s dbgMat.getNumElements():%d\n", _name.c_str(),
				dbgMat.getNumElements());
		float meanOptimWC = 0.5f * dbgMat.sum()
				/ (float) (dbgMat.getNumElements());
		printf("FCLayer: %s bias meanOptimWC:%.8f\n", _name.c_str(),
				meanOptimWC);
	}

}

void FCLayer::bpropWeights(NVMatrix& v, int inpIdx, PASS_TYPE passType) {
	int numCases = v.getNumRows();

	NVMatrix& prevActs_T = _prev[inpIdx]->getActs().getTranspose();
	float scaleInc = (_weights[inpIdx].getNumUpdates() == 0
			&& passType != PASS_GC) * _weights[inpIdx].getMom();
	float scaleGrad =
			passType == PASS_GC ? 1 : _weights[inpIdx].getEps() / numCases;

	_weights[inpIdx].getInc().addProduct(prevActs_T, v, scaleInc, scaleGrad);

	if (0) {
		// debugging: print gradient wrt weights
		NVMatrix dbgMat(_weights[inpIdx].getInc());
		dbgMat.addProduct(prevActs_T, v, 0, scaleGrad);
		dbgMat.eltwiseDivide(_weights[inpIdx].getW());
		dbgMat.apply(NVMatrixOps::Abs());
		float meanOptimWC = 0.5f * dbgMat.sum()
				/ (float) (dbgMat.getNumElements());
		printf("FCLayer: %s weights meanOptimWC:%.8f\n", _name.c_str(),
				meanOptimWC);
	}

	delete &prevActs_T;
}

/* 
 * =======================
 * LocalLayer
 * =======================
 */
LocalLayer::LocalLayer(ConvNet* convNet, PyObject* paramsDict, bool useGrad) :
		WeightLayer(convNet, paramsDict, false, useGrad) {
	_padding = pyDictGetIntV(paramsDict, "padding");
	_stride = pyDictGetIntV(paramsDict, "stride");
	_filterSize = pyDictGetIntV(paramsDict, "filterSize");
	_channels = pyDictGetIntV(paramsDict, "channels");
	_imgSize = pyDictGetIntV(paramsDict, "imgSize");
	_numFilters = pyDictGetInt(paramsDict, "filters");
	_groups = pyDictGetIntV(paramsDict, "groups");
	_filterChannels = pyDictGetIntV(paramsDict, "filterChannels");
	_randSparse = pyDictGetIntV(paramsDict, "randSparse");
	_overSample = pyDictGetIntV(paramsDict, "overSample");
	_filterPixels = pyDictGetIntV(paramsDict, "filterPixels");
	_imgPixels = pyDictGetIntV(paramsDict, "imgPixels");

	_modulesX = pyDictGetInt(paramsDict, "modulesX");
	_modules = pyDictGetInt(paramsDict, "modules");

	// It's a vector on the heap to be consistent with all the others...
	_filterConns = new vector<FilterConns>();
	PyObject* pyFilterConns = PyDict_GetItemString(paramsDict, "filterConns");
	for (int i = 0; i < _randSparse->size(); i++) {
		FilterConns fc;
		if (_randSparse->at(i)) {
			fc.hFilterConns = getIntA(PyList_GET_ITEM(pyFilterConns, i));
		}
		_filterConns->push_back(fc);
	}
}

void LocalLayer::copyToGPU() {
	WeightLayer::copyToGPU();
	for (int i = 0; i < _prev.size(); i++) {
		if (_randSparse->at(i)) { // Copy to GPU vector that describes sparse random connectivity
			cudaMalloc(&_filterConns->at(i).dFilterConns,
					sizeof(int) * _groups->at(i) * _filterChannels->at(i));
			gpuMonitor->addUsedMemory(
					sizeof(int) * _groups->at(i) * _filterChannels->at(i));
			cudaMemcpy(_filterConns->at(i).dFilterConns,
					_filterConns->at(i).hFilterConns,
					sizeof(int) * _groups->at(i) * _filterChannels->at(i),
					cudaMemcpyHostToDevice);
			cutilCheckMsg("cudaMemcpy: failed");
		}
	}
}

/* 
 * =======================
 * ConvLayer
 * =======================
 */
ConvLayer::ConvLayer(ConvNet* convNet, PyObject* paramsDict) :
		LocalLayer(convNet, paramsDict, true) {
	_partialSum = pyDictGetInt(paramsDict, "partialSum");
	_sharedBiases = pyDictGetInt(paramsDict, "sharedBiases");
}

void ConvLayer::fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType) {
//	printf("ConvLayer %s. fpropActs\n",_name.c_str());
	checkNVMatrixNan(*_weights[inpIdx], _name + string("weights"));
	checkNVMatrixNan(*_inputs[inpIdx], _name + string("inputs"));
	if (_randSparse->at(inpIdx)) {
		convFilterActsSparse(*_inputs[inpIdx], *_weights[inpIdx], getActs(),
				_filterConns->at(inpIdx).dFilterConns, _imgSize->at(inpIdx),
				_modulesX, _modulesX, _padding->at(inpIdx), _stride->at(inpIdx),
				_channels->at(inpIdx), _filterChannels->at(inpIdx),
				_groups->at(inpIdx), scaleTargets, 1);
	} else {
		convFilterActs(*_inputs[inpIdx], *_weights[inpIdx], getActs(),
				_imgSize->at(inpIdx), _modulesX, _modulesX,
				_padding->at(inpIdx), _stride->at(inpIdx),
				_channels->at(inpIdx), _groups->at(inpIdx), scaleTargets, 1);
	}

	if (scaleTargets == 0) {
		if (_sharedBiases) {
			getActs().reshape(_numFilters,
					getActs().getNumElements() / _numFilters);
			getActs().addVector(_biases->getW());
			getActs().reshape(_numFilters * _modules,
					getActs().getNumElements() / (_numFilters * _modules));
		} else {
			getActs().addVector(_biases->getW());
		}
	}
	//if(verbose){
	//	NVMatrix tmp(getActs());
	//	getActs().apply(NVMatrixOps::Abs(),tmp);
	//	float mean_abs_act = tmp.sum()/tmp.getNumElements();
	//	printf("ConvLayer %s. mean_abs_act:%f\n",_name.c_str(),mean_abs_act);
	//}

//	if (isnan(getActs().sum())) {
//		printf("ConvLayer::fpropActs\n");
//		printf("inputs min:%f max:%f mean:%f\n", (*_inputs[inpIdx]).min(),
//				(*_inputs[inpIdx]).max(), (*_inputs[inpIdx]).mean());
//		printf("(*_weights[inpIdx]) min:%f max:%f mean:%f\n",
//				(*_weights[inpIdx]).min(), (*_weights[inpIdx]).max(),
//				(*_weights[inpIdx]).mean());
//		printf("_biases min:%f max:%f mean:%f\n", (_biases->getW()).min(),
//				(_biases->getW()).max(), (_biases->getW()).mean());
//		(*_inputs[inpIdx]).fprint(
//				(_name + string("ConvLayer fpropActs  _inputs")).c_str(),
//				(*_inputs[inpIdx]).getNumRows(),
//				(*_inputs[inpIdx]).getNumCols());
//		(*_weights[inpIdx]).fprint(
//				(_name + string("ConvLayer fpropActs   _weights")).c_str(),
//				(*_weights[inpIdx]).getNumRows(),
//				(*_weights[inpIdx]).getNumCols());
//		(_biases->getW()).fprint(
//				(_name + string("ConvLayer fpropActs  _biases")).c_str(),
//				(_biases->getW()).getNumRows(), (_biases->getW()).getNumCols());
//	}
//
//	checkNVMatrixNan(getActs(), _name);
}

void ConvLayer::bpropBiases(NVMatrix& v, PASS_TYPE passType) {
	int numCases = v.getNumCols();
	float scaleBGrad = passType == PASS_GC ? 1 : _biases->getEps() / numCases;
	if (_sharedBiases) {
		v.reshape(_numFilters, v.getNumElements() / _numFilters);
		_biases->getGrad().addSum(v, 1, 0, scaleBGrad);
		v.reshape(_numFilters * _modules,
				v.getNumElements() / (_numFilters * _modules));
		if (0) {
			NVMatrix dbgMat(_biases->getGrad(), true);
			dbgMat.eltwiseDivide(_biases->getW());
			dbgMat.apply(NVMatrixOps::Abs());
			float meanOptimWC = 0.5f * dbgMat.sum()
					/ (float) (dbgMat.getNumElements());
			printf("ConvLayer: %s biases meanOptimWC:%.8f\n", _name.c_str(),
					meanOptimWC);
		}
	} else {
		_biases->getGrad().addSum(v, 1, 0, scaleBGrad);
	}
}

void ConvLayer::bpropWeights(NVMatrix& v, int inpIdx, PASS_TYPE passType) {
	int numCases = v.getNumCols();

	NVMatrix& tgt =
			_partialSum > 0 ? _weightGradTmp : _weights[inpIdx].getGrad();
	float scaleWGrad =
			passType == PASS_GC ? 1 : _weights[inpIdx].getEps() / numCases;
	float scaleTargets = _weights[inpIdx].getNumUpdates() > 0
			&& _partialSum == 0; // ? 1 : 0;
	if (_randSparse->at(inpIdx)) {
		convWeightActsSparse(_prev[inpIdx]->getActs(), v, tgt,
				_filterConns->at(inpIdx).dFilterConns, _imgSize->at(inpIdx),
				_modulesX, _modulesX, _filterSize->at(inpIdx),
				_padding->at(inpIdx), _stride->at(inpIdx),
				_channels->at(inpIdx), _filterChannels->at(inpIdx),
				_groups->at(inpIdx), _partialSum, scaleTargets, scaleWGrad);
	} else {
		convWeightActs(_prev[inpIdx]->getActs(), v, tgt, _imgSize->at(inpIdx),
				_modulesX, _modulesX, _filterSize->at(inpIdx),
				_padding->at(inpIdx), _stride->at(inpIdx),
				_channels->at(inpIdx), _groups->at(inpIdx), _partialSum,
				scaleTargets, scaleWGrad);
	}
	if (_partialSum > 0) {
		scaleTargets = _weights[inpIdx].getNumUpdates() > 0;
		_weightGradTmp.reshape(_modules / _partialSum,
				_filterChannels->at(inpIdx) * _filterPixels->at(inpIdx)
						* _numFilters);
		_weights[inpIdx].getGrad().addSum(_weightGradTmp, 0, scaleTargets, 1);
		_weights[inpIdx].getGrad().reshape(
				_filterChannels->at(inpIdx) * _filterPixels->at(inpIdx),
				_numFilters);
		if (0) {
			NVMatrix dbgMat(_weights[inpIdx].getGrad(), true);
			dbgMat.eltwiseDivide(_weights[inpIdx].getW());
			dbgMat.apply(NVMatrixOps::Abs());
			float meanOptimWC = 0.5f * dbgMat.sum()
					/ (float) (dbgMat.getNumElements());
			printf("ConvLayer: %s weights meanOptimWC:%.8f\n", _name.c_str(),
					meanOptimWC);
		}
	}
}

void ConvLayer::bpropActs(NVMatrix& v, int inpIdx, float scaleTargets,
		PASS_TYPE passType) {
	if (_randSparse->at(inpIdx)) {
		NVMatrix& tgt =
				_overSample->at(inpIdx) > 1 ?
						_actGradTmp : _prev[inpIdx]->getActsGrad();
		convImgActsSparse(v, *_weights[inpIdx], tgt,
				_filterConns->at(inpIdx).dFilterConns, _imgSize->at(inpIdx),
				_imgSize->at(inpIdx), _modulesX, _padding->at(inpIdx),
				_stride->at(inpIdx), _channels->at(inpIdx),
				_filterChannels->at(inpIdx), _groups->at(inpIdx), scaleTargets,
				1);
		if (_overSample->at(inpIdx) > 1) {
			_actGradTmp.reshape(_overSample->at(inpIdx),
					_actGradTmp.getNumElements() / _overSample->at(inpIdx));
			_actGradTmp.sum(0, _prev[inpIdx]->getActsGrad());
			_prev[inpIdx]->getActsGrad().reshape(
					_prev[inpIdx]->getActsGrad().getNumElements()
							/ v.getNumCols(), v.getNumCols());
		}
	} else {
		convImgActs(v, *_weights[inpIdx], _prev[inpIdx]->getActsGrad(),
				_imgSize->at(inpIdx), _imgSize->at(inpIdx), _modulesX,
				_padding->at(inpIdx), _stride->at(inpIdx),
				_channels->at(inpIdx), _groups->at(inpIdx), scaleTargets, 1);
		//checkNVMatrixNan(_prev[inpIdx]->getActsGrad(), _name);
	}
}

void ConvLayer::truncBwdActs() {
	LocalLayer::truncBwdActs();
	if (_conserveMem) {
		_weightGradTmp.truncate();
		_actGradTmp.truncate();
	}
}

void ConvLayer::fpropPostCommon(NVMatrixV& v, PASS_TYPE passType) {
	Layer::fpropPostCommon(v, passType);
	//float actMax=getActs().max(),actMin=getActs().min();
	//if(abs(actMax)>thres || abs(actMin)>thres){
	//	float wMin,wMax,wMean;
	//	wMin=(*_weights[0]).min();
	//	wMax=(*_weights[0]).max();
	//	wMean=(*_weights[0]).mean();
	//	printf("layer:%s wMin,wMax,wMean:%f %f %f\n",
	//		_name.c_str(),wMin,wMax,wMean);
	//}
}
/* 
 * =======================
 * LocalUnsharedLayer
 * =======================
 */
LocalUnsharedLayer::LocalUnsharedLayer(ConvNet* convNet, PyObject* paramsDict) :
		LocalLayer(convNet, paramsDict, false) {
}

void LocalUnsharedLayer::fpropActs(int inpIdx, float scaleTargets,
		PASS_TYPE passType) {
	if (_randSparse->at(inpIdx)) {
		localFilterActsSparse(*_inputs[inpIdx], *_weights[inpIdx], getActs(),
				_filterConns->at(inpIdx).dFilterConns, _imgSize->at(inpIdx),
				_modulesX, _modulesX, _padding->at(inpIdx), _stride->at(inpIdx),
				_channels->at(inpIdx), _filterChannels->at(inpIdx),
				_groups->at(inpIdx), scaleTargets, 1);
	} else {
		localFilterActs(*_inputs[inpIdx], *_weights[inpIdx], getActs(),
				_imgSize->at(inpIdx), _modulesX, _modulesX,
				_padding->at(inpIdx), _stride->at(inpIdx),
				_channels->at(inpIdx), _groups->at(inpIdx), scaleTargets, 1);

	}
	if (scaleTargets == 0) {
		getActs().addVector(_biases->getW());
	}
}

void LocalUnsharedLayer::bpropBiases(NVMatrix& v, PASS_TYPE passType) {
	int numCases = v.getNumCols();
	float scaleBGrad = passType == PASS_GC ? 1 : _biases->getEps() / numCases;
	_biases->getGrad().addSum(v, 1, 0, scaleBGrad);
}

void LocalUnsharedLayer::bpropWeights(NVMatrix& v, int inpIdx,
		PASS_TYPE passType) {
	int numCases = v.getNumCols();

	float scaleInc = (passType != PASS_GC
			&& _weights[inpIdx].getNumUpdates() == 0)
			* _weights[inpIdx].getMom(); // momentum
	float scaleWGrad =
			passType == PASS_GC ? 1 : _weights[inpIdx].getEps() / numCases; // eps / numCases
	if (_randSparse->at(inpIdx)) {
		localWeightActsSparse(_prev[inpIdx]->getActs(), v,
				_weights[inpIdx].getInc(),
				_filterConns->at(inpIdx).dFilterConns, _imgSize->at(inpIdx),
				_modulesX, _modulesX, _filterSize->at(inpIdx),
				_padding->at(inpIdx), _stride->at(inpIdx),
				_channels->at(inpIdx), _filterChannels->at(inpIdx),
				_groups->at(inpIdx), scaleInc, scaleWGrad);
	} else {
		localWeightActs(_prev[inpIdx]->getActs(), v, _weights[inpIdx].getInc(),
				_imgSize->at(inpIdx), _modulesX, _modulesX,
				_filterSize->at(inpIdx), _padding->at(inpIdx),
				_stride->at(inpIdx), _channels->at(inpIdx), _groups->at(inpIdx),
				scaleInc, scaleWGrad);
	}
}

void LocalUnsharedLayer::bpropActs(NVMatrix& v, int inpIdx, float scaleTargets,
		PASS_TYPE passType) {
	if (_randSparse->at(inpIdx)) {
		localImgActsSparse(v, *_weights[inpIdx], _prev[inpIdx]->getActsGrad(),
				_filterConns->at(inpIdx).dFilterConns, _imgSize->at(inpIdx),
				_imgSize->at(inpIdx), _modulesX, _padding->at(inpIdx),
				_stride->at(inpIdx), _channels->at(inpIdx),
				_filterChannels->at(inpIdx), _groups->at(inpIdx), scaleTargets,
				1);
	} else {
		localImgActs(v, *_weights[inpIdx], _prev[inpIdx]->getActsGrad(),
				_imgSize->at(inpIdx), _imgSize->at(inpIdx), _modulesX,
				_padding->at(inpIdx), _stride->at(inpIdx),
				_channels->at(inpIdx), _groups->at(inpIdx), scaleTargets, 1);
	}
}

/* 
 * =======================
 * SoftmaxLayer
 * =======================
 */
SoftmaxLayer::SoftmaxLayer(ConvNet* convNet, PyObject* paramsDict) :
		Layer(convNet, paramsDict, true) {
}

void SoftmaxLayer::fpropActs(int inpIdx, float scaleTargets,
		PASS_TYPE passType) {
	NVMatrix& input = *_inputs[0];
	NVMatrix& max = input.max(1);
	input.addVector(max, -1, getActs());

	//Matrix hMax;
	//max.copyToHost(hMax,true);
	//float maxmax=hMax.max();

	//Matrix hActMat;
	//getActs().copyToHost(hActMat,true);
	//float actmin=hActMat.min();

	getActs().apply(NVMatrixOps::Exp_a());

	//getActs().copyToHost(hActMat,true);
	//if(hActMat.min()<=0)
	//{
	//	printf("act min before exp:%10.9f\n",actmin);
	//	printf("after exp hActMat.min()<=0  %10.9f\n",hActMat.min());
	//}

	assert(getActs().isContiguous());

	//   Matrix hMat;
	//   getActs().copyToHost(hMat,true);

	//printf("SoftmaxLayer::fpropActs: check _outputs after do exponentiation\n");
	//   for(int p=0,i=0;i<hMat.getNumCols();++i){
	//   	for(int j=0;j<hMat.getNumRows();++j,++p){
	//   		if(hMat.getCell(j,i)<=0){
	//
	//   			printf("nonpositive element is found. %6.5f (%d,%d)\n",hMat.getCell(j,i),j,i);
	//			exit(1);
	//   		}
	//   	}
	//   }

	NVMatrix& sum = getActs().sum(1);
	getActs().eltwiseDivideByVector(sum);

	//getActs().copyToHost(hMat,true);

	//for(int p=0,i=0;i<hMat.getNumCols();++i){
	//	for(int j=0;j<hMat.getNumRows();++j,++p){
	//		if(isnan(hMat.getCell(j,i)) || hMat.getCell(j,i)<=0){
	//			printf("SoftmaxLayer::fpropActs: check _outputs after getting probability\n");
	//			printf("nan/nonpositive element is found. %7.6f (%d,%d)\n",hMat.getCell(j,i),j,i);
	//		}
	//	}
	//}

	delete &max;
	delete &sum;
}

void SoftmaxLayer::bpropActs(NVMatrix& v, int inpIdx, float scaleTargets,
		PASS_TYPE passType) {
	assert(inpIdx == 0);
	bool doLogregGrad = _next.size() == 1
			&& _next[0]->getType() == "cost.logreg";
	if (doLogregGrad) {
		NVMatrix& labels = _next[0]->getPrev()[0]->getActs();
		float gradCoeff = dynamic_cast<CostLayer*>(_next[0])->getCoeff();
		computeLogregSoftmaxGrad(labels, getActs(), _prev[0]->getActsGrad(),
				scaleTargets == 1, gradCoeff);
	} else {
		computeSoftmaxGrad(getActs(), v, _prev[0]->getActsGrad(),
				scaleTargets == 1);
	}
}

/* 
 * =======================
 * LogSoftmaxLayer
 * =======================
 */
LogSoftmaxLayer::LogSoftmaxLayer(ConvNet* convNet, PyObject* paramsDict) :
		Layer(convNet, paramsDict, true) {
}

void LogSoftmaxLayer::fpropActs(int inpIdx, float scaleTargets,
		PASS_TYPE passType) {
	NVMatrix& input = *_inputs[0];
	NVMatrix& max = input.max(1);
//	checkNVMatrixNan(input, _name);
//	checkNVMatrixNan(max, _name);

	input.addVector(max, -1, getActs());
//	checkNVMatrixNan(getActs(), _name);

	NVMatrix sumExp(getActs(), true);
	sumExp.apply(NVMatrixOps::Exp_a());
	assert(sumExp.isContiguous());
	NVMatrix& sum = sumExp.sum(1);
	sum.apply(NVMatrixOps::Log());

//	checkNVMatrixNan(sum, _name);
	getActs().addVector(sum, -1);
//	checkNVMatrixNan(getActs(), _name);
	delete &max;
	delete &sum;
}

void LogSoftmaxLayer::bpropActs(NVMatrix& v, int inpIdx, float scaleTargets,
		PASS_TYPE passType) {
	assert(inpIdx == 0);
	bool doLogregGrad = _next.size() == 1 && _next[0]->getType() == "cost.reg";
	if (doLogregGrad) {
		//printf("LogSoftmaxLayer::bpropActs doLogregGrad\n");
		NVMatrix& labels = _next[0]->getPrev()[0]->getActs();
		float gradCoeff = dynamic_cast<CostLayer*>(_next[0])->getCoeff();
		computeLogregSoftmaxGrad(labels, getActs(), _prev[0]->getActsGrad(),
				scaleTargets == 1, gradCoeff, true);
	} else {
		// to do
		computeSoftmaxGrad(getActs(), v, _prev[0]->getActsGrad(),
				scaleTargets == 1);
	}
	checkNVMatrixNan(_prev[0]->getActsGrad(), _name);
}

/*
 * =======================
 * EltwiseSumLayer
 * =======================
 */
EltwiseSumLayer::EltwiseSumLayer(ConvNet* convNet, PyObject* paramsDict) :
		Layer(convNet, paramsDict, false) {
	_coeffs = pyDictGetFloatV(paramsDict, "coeffs");
}

void EltwiseSumLayer::fpropActs(int inpIdx, float scaleTargets,
		PASS_TYPE passType) {
	if (scaleTargets == 0) {
		_inputs[inpIdx]->scale(_coeffs->at(inpIdx), getActs());
//		printf("EltwiseSumLayer::fpropActs scaleTargets==0 coeff:%f prev layer name:%s\n",
//				_coeffs->at(inpIdx),_prev[inpIdx]->getName().c_str());
//		getActs().print(getActs().getNumRows(),2);
	} else {
		getActs().add(*_inputs[inpIdx], _coeffs->at(inpIdx));
//		printf("EltwiseSumLayer::fpropActs scaleTargets!=0 coeff:%f prev layer name:%s\n",
//				_coeffs->at(inpIdx),_prev[inpIdx]->getName().c_str());
//		printf("inputs\n");
//		_inputs[inpIdx]->print(_inputs[inpIdx]->getNumRows(),2);
//		printf("getActs()\n");
//		getActs().print(getActs().getNumRows(),2);
//		exit(0);
	}
}

void EltwiseSumLayer::bpropActs(NVMatrix& v, int inpIdx, float scaleTargets,
		PASS_TYPE passType) {
	if (scaleTargets == 0) {
		v.scale(_coeffs->at(inpIdx), _prev[inpIdx]->getActsGrad());
	} else {
		assert(&_prev[inpIdx]->getActsGrad() != &v);
		_prev[inpIdx]->getActsGrad().add(v, scaleTargets, _coeffs->at(inpIdx));
	}
	//printf("EltwiseSumLayer::bpropActs %s. _prev[inpIdx]->getActsGrad() rows,cols:%d %d inpIdx:%d coeff:%.3f\n",
	//	_name.c_str(),_prev[inpIdx]->getActsGrad().getNumRows(),
	//	_prev[inpIdx]->getActsGrad().getNumCols(),inpIdx,_coeffs->at(inpIdx));
}

/* 
 * =======================
 * EltwiseMaxLayer
 * =======================
 */
EltwiseMaxLayer::EltwiseMaxLayer(ConvNet* convNet, PyObject* paramsDict) :
		Layer(convNet, paramsDict, false) {
}

void EltwiseMaxLayer::fpropActs(int inpIdx, float scaleTargets,
		PASS_TYPE passType) {
	if (inpIdx == 1) { // First input, do nothing
		_inputs[inpIdx]->applyBinary(NVMatrixAggs::Max(), *_inputs[0],
				getActs());
	} else if (inpIdx > 1) {
		getActs().applyBinary(NVMatrixAggs::Max(), *_inputs[inpIdx]);
	}
}

void EltwiseMaxLayer::bpropActs(NVMatrix& v, int inpIdx, float scaleTargets,
		PASS_TYPE passType) {
	computeEltwiseMaxGrad(v, *_inputs[inpIdx], getActs(),
			_prev[inpIdx]->getActsGrad(), scaleTargets != 0);
}

/* 
 * =======================
 * DataLayer
 * =======================
 */
DataLayer::DataLayer(ConvNet* convNet, PyObject* paramsDict) :
		Layer(convNet, paramsDict, false) {
	_dataIdx = pyDictGetInt(paramsDict, "dataIdx");
}

void DataLayer::fprop(PASS_TYPE passType) {
	throw string("No dava given!");
}

void DataLayer::fpropActs(int inpIdx, float scaleTargets, PASS_TYPE passType) {
}

void DataLayer::fprop(NVMatrixV& data, PASS_TYPE passType) {
	_outputs = data[_dataIdx];
	checkNVMatrixNan(*_outputs, _name);
#ifdef MULTIGPU
	NVMatrix::checkCUDAError(cudaEventRecord(_fpropEvent));
#endif
	fpropNext(passType);
}

bool DataLayer::isGradProducer() {
	return false;
}

/* 
 * =====================
 * PoolLayer
 * =====================
 */
PoolLayer::PoolLayer(ConvNet* convNet, PyObject* paramsDict, bool trans) :
		Layer(convNet, paramsDict, trans) {
	_channels = pyDictGetInt(paramsDict, "channels");
	_sizeX = pyDictGetInt(paramsDict, "sizeX");
	_start = pyDictGetInt(paramsDict, "start");
	_stride = pyDictGetInt(paramsDict, "stride");
	_outputsX = pyDictGetInt(paramsDict, "outputsX");
	_imgSize = pyDictGetInt(paramsDict, "imgSize");
	_pool = pyDictGetString(paramsDict, "pool");
}

PoolLayer& PoolLayer::makePoolLayer(ConvNet* convNet, PyObject* paramsDict) {
	string _pool = pyDictGetString(paramsDict, "pool");
	if (_pool == "max") {
		return *new MaxPoolLayer(convNet, paramsDict);
	} else if (_pool == "avg") {
		return *new AvgPoolLayer(convNet, paramsDict);
	}
	throw string("Unknown pooling layer type ") + _pool;
}

/* 
 * =====================
 * AvgPoolLayer
 * =====================
 */
AvgPoolLayer::AvgPoolLayer(ConvNet* convNet, PyObject* paramsDict) :
		PoolLayer(convNet, paramsDict, false) {
}

void AvgPoolLayer::fpropActs(int inpIdx, float scaleTargets,
		PASS_TYPE passType) {
	convLocalPool(*_inputs[0], getActs(), _channels, _sizeX, _start, _stride,
			_outputsX, AvgPooler());
}

void AvgPoolLayer::bpropActs(NVMatrix& v, int inpIdx, float scaleTargets,
		PASS_TYPE passType) {
	convLocalAvgUndo(v, _prev[0]->getActsGrad(), _sizeX, _start, _stride,
			_outputsX, _imgSize, scaleTargets, 1);
}

/* 
 * =====================
 * MaxPoolLayer
 * =====================
 */
MaxPoolLayer::MaxPoolLayer(ConvNet* convNet, PyObject* paramsDict) :
		PoolLayer(convNet, paramsDict, false) {
}

void MaxPoolLayer::fpropActs(int inpIdx, float scaleTargets,
		PASS_TYPE passType) {
	convLocalPool(*_inputs[0], getActs(), _channels, _sizeX, _start, _stride,
			_outputsX, MaxPooler());

	checkNVMatrixNan(getActs(), _name);
}

void MaxPoolLayer::bpropActs(NVMatrix& v, int inpIdx, float scaleTargets,
		PASS_TYPE passType) {
	convLocalMaxUndo(_prev[0]->getActs(), v, getActs(),
			_prev[inpIdx]->getActsGrad(), _sizeX, _start, _stride, _outputsX,
			scaleTargets, 1);
	checkNVMatrixNan(_prev[inpIdx]->getActsGrad(), _name);
}

/* 
 * =====================
 * NailbedLayer
 * =====================
 */
NailbedLayer::NailbedLayer(ConvNet* convNet, PyObject* paramsDict) :
		Layer(convNet, paramsDict, false) {
	_channels = pyDictGetInt(paramsDict, "channels");
	_start = pyDictGetInt(paramsDict, "start");
	_stride = pyDictGetInt(paramsDict, "stride");
	_outputsX = pyDictGetInt(paramsDict, "outputsX");
	_imgSize = pyDictGetInt(paramsDict, "imgSize");
}

void NailbedLayer::fpropActs(int inpIdx, float scaleTargets,
		PASS_TYPE passType) {
	convBedOfNails(*_inputs[0], getActs(), _channels, _imgSize, _start, _stride,
			0, 1);
}

void NailbedLayer::bpropActs(NVMatrix& v, int inpIdx, float scaleTargets,
		PASS_TYPE passType) {
	convBedOfNailsUndo(v, _prev[0]->getActsGrad(), _channels, _imgSize, _start,
			_stride, scaleTargets, 1);
}

/* 
 * =====================
 * GaussianBlurLayer
 * =====================
 */
GaussianBlurLayer::GaussianBlurLayer(ConvNet* convNet, PyObject* paramsDict) :
		Layer(convNet, paramsDict, false) {
	_channels = pyDictGetInt(paramsDict, "channels");
	_hFilter = pyDictGetMatrix(paramsDict, "filter");
}

void GaussianBlurLayer::fpropActs(int inpIdx, float scaleTargets,
		PASS_TYPE passType) {
	convGaussianBlur(*_inputs[0], _filter, getActs(), true, _channels, 0, 1);
	convGaussianBlur(getActs(), _filter, getActs(), false, _channels, 0, 1);
}

// This is here just for completeness' sake. Why would you backpropagate
// through a blur filter?
void GaussianBlurLayer::bpropActs(NVMatrix& v, int inpIdx, float scaleTargets,
		PASS_TYPE passType) {
	NVMatrix& tgt1 =
			_prev[0]->getRcvdBInputs() > 0 ?
					_actGradsTmp : _prev[0]->getActsGrad();
	convGaussianBlur(v, _filter, tgt1, true, _channels, 0, 1);
	convGaussianBlur(tgt1, _filter, _prev[0]->getActsGrad(), false, _channels,
			scaleTargets, 1);
}

void GaussianBlurLayer::copyToGPU() {
	Layer::copyToGPU();
	_filter.copyFromHost(*_hFilter, true);
}

/* 
 * =====================
 * ResizeLayer
 * =====================
 */
ResizeLayer::ResizeLayer(ConvNet* convNet, PyObject* paramsDict) :
		Layer(convNet, paramsDict, false) {
	_channels = pyDictGetInt(paramsDict, "channels");
	_imgSize = pyDictGetInt(paramsDict, "imgSize");
	_tgtSize = pyDictGetInt(paramsDict, "tgtSize");
	_scale = pyDictGetFloat(paramsDict, "scale");
}

void ResizeLayer::fpropActs(int inpIdx, float scaleTargets,
		PASS_TYPE passType) {
	convResizeBilinear(*_inputs[0], getActs(), _imgSize, _tgtSize, _scale);
}

// Can't do this
void ResizeLayer::bpropActs(NVMatrix& v, int inpIdx, float scaleTargets,
		PASS_TYPE passType) {
	assert(false);
}

/* 
 * =====================
 * RGBToYUVLayer
 * =====================
 */
RGBToYUVLayer::RGBToYUVLayer(ConvNet* convNet, PyObject* paramsDict) :
		Layer(convNet, paramsDict, false) {
}

void RGBToYUVLayer::fpropActs(int inpIdx, float scaleTargets,
		PASS_TYPE passType) {
	convRGBToYUV(*_inputs[0], getActs());
}

// Can't do this
void RGBToYUVLayer::bpropActs(NVMatrix& v, int inpIdx, float scaleTargets,
		PASS_TYPE passType) {
	assert(false);
}

/* 
 * =====================
 * RGBToLABLayer
 * =====================
 */
RGBToLABLayer::RGBToLABLayer(ConvNet* convNet, PyObject* paramsDict) :
		Layer(convNet, paramsDict, false) {
	_center = pyDictGetInt(paramsDict, "center");
}

void RGBToLABLayer::fpropActs(int inpIdx, float scaleTargets,
		PASS_TYPE passType) {
	convRGBToLAB(*_inputs[0], getActs(), _center);
}

// Can't do this
void RGBToLABLayer::bpropActs(NVMatrix& v, int inpIdx, float scaleTargets,
		PASS_TYPE passType) {
	assert(false);
}

/* 
 * =====================
 * ResponseNormLayer
 * =====================
 */
ResponseNormLayer::ResponseNormLayer(ConvNet* convNet, PyObject* paramsDict) :
		Layer(convNet, paramsDict, false) {
	_channels = pyDictGetInt(paramsDict, "channels");
	_size = pyDictGetInt(paramsDict, "size");

	_bias = pyDictGetFloat(paramsDict, "bias");
	_scale = pyDictGetFloat(paramsDict, "scale");
	_pow = pyDictGetFloat(paramsDict, "pow");
	printf("(bias,scale,pow)=(%f,%f,%f)\n", _bias, _scale, _pow);
}

void ResponseNormLayer::fpropActs(int inpIdx, float scaleTargets,
		PASS_TYPE passType) {
	convResponseNorm(*_inputs[0], _denoms, getActs(), _channels, _size, _scale,
			_pow);
}

void ResponseNormLayer::bpropActs(NVMatrix& v, int inpIdx, float scaleTargets,
		PASS_TYPE passType) {
	convResponseNormUndo(v, _denoms, _prev[0]->getActs(), getActs(),
			_prev[0]->getActsGrad(), _channels, _size, _scale, _pow,
			scaleTargets, 1);
}

void ResponseNormLayer::truncBwdActs() {
	Layer::truncBwdActs();
	if (_conserveMem) {
		_denoms.truncate();
	}
}

/* 
 * =====================
 * CrossMapResponseNormLayer
 * =====================
 */
CrossMapResponseNormLayer::CrossMapResponseNormLayer(ConvNet* convNet,
		PyObject* paramsDict) :
		ResponseNormLayer(convNet, paramsDict) {
	_blocked = pyDictGetInt(paramsDict, "blocked");
}

void CrossMapResponseNormLayer::fpropActs(int inpIdx, float scaleTargets,
		PASS_TYPE passType) {
	convResponseNormCrossMap(*_inputs[0], _denoms, getActs(), _channels, _size,
			_bias, _scale, _pow, _blocked);
	checkNVMatrixNan(getActs(), _name);
}

void CrossMapResponseNormLayer::bpropActs(NVMatrix& v, int inpIdx,
		float scaleTargets, PASS_TYPE passType) {
	convResponseNormCrossMapUndo(v, _denoms, _prev[0]->getActs(), getActs(),
			_prev[0]->getActsGrad(), _channels, _size, _bias, _scale, _pow,
			_blocked, scaleTargets, 1);
	checkNVMatrixNan(_prev[0]->getActsGrad(), _name);
}

/* 
 * =====================
 * ContrastNormLayer
 * =====================
 */
ContrastNormLayer::ContrastNormLayer(ConvNet* convNet, PyObject* paramsDict) :
		ResponseNormLayer(convNet, paramsDict) {
	_imgSize = pyDictGetInt(paramsDict, "imgSize");
}

void ContrastNormLayer::fpropActs(int inpIdx, float scaleTargets,
		PASS_TYPE passType) {
	NVMatrix& images = *_inputs[0];
	convLocalPool(images, _meanDiffs, _channels, _size, -_size / 2, 1, _imgSize,
			AvgPooler());
	_meanDiffs.add(images, -1, 1);
	convContrastNorm(images, _meanDiffs, _denoms, getActs(), _channels, _size,
			_scale, _pow);
}

void ContrastNormLayer::bpropActs(NVMatrix& v, int inpIdx, float scaleTargets,
		PASS_TYPE passType) {
	convContrastNormUndo(v, _denoms, _meanDiffs, getActs(),
			_prev[inpIdx]->getActsGrad(), _channels, _size, _scale, _pow,
			scaleTargets, 1);
}

void ContrastNormLayer::truncBwdActs() {
	ResponseNormLayer::truncBwdActs();
	if (_conserveMem) {
		_meanDiffs.truncate();
	}
}

/* 
 * =====================
 * CostLayer
 * =====================
 */
CostLayer::CostLayer(ConvNet* convNet, PyObject* paramsDict, bool trans) :
		Layer(convNet, paramsDict, trans) {
	_coeff = pyDictGetFloat(paramsDict, "coeff");
}

float CostLayer::getCoeff() {
	return _coeff;
}

void CostLayer::setCoeff(float coeff) {
	assert(coeff > 0);
	_coeff = coeff;
}

void CostLayer::scaleCoeff(float scale) {
	assert(scale > 0);
	float oldCoeff = _coeff;
	_coeff *= scale;
	printf(
			"===========Cost Layer %s. Coeffient is scaled by %3.2f. %6.5f->%6.5f=====\n",
			_name.c_str(), scale, oldCoeff, _coeff);
}

void CostLayer::bprop(PASS_TYPE passType) {
	if (_coeff != 0) {
		Layer::bprop(passType);
	}
}

bool CostLayer::isGradProducer() {
	return _coeff != 0;
}

doublev& CostLayer::getCost() {
	doublev& v = *new doublev();
	v.insert(v.begin(), _costv.begin(), _costv.end());
	return v;
}

CostLayer& CostLayer::makeCostLayer(ConvNet* convNet, string& type,
		PyObject* paramsDict) {
	if (type == "cost.logreg") {
		return *new LogregCostLayer(convNet, paramsDict);
	} else if (type == "cost.sum2") {
		return *new SumOfSquaresCostLayer(convNet, paramsDict);
	} else if (type == "cost.reg") {
		return *new RegCostLayer(convNet, paramsDict);
	} else if (type == "cost.sumsquaresdiff") {
		return *new SumOfSquaresOfDiffCostLayer(convNet, paramsDict);
	} else if (type == "cost.logsumsquaresdiff") {
		return *new LogSumOfSquaresOfDiffCostLayer(convNet, paramsDict);
	} else if (type == "cost.color_enhance") {
		return *new ColorEnhanceCostLayer(convNet, paramsDict);
	} else if (type == "cost.color_enhance_separate") {
		return *new ColorEnhanceSeparateCostLayer(convNet, paramsDict);
	} else if (type == "cost.gradmag_enhance") {
		return *new GradMagEnhanceCostLayer(convNet, paramsDict);
	}
	throw string("Unknown cost layer type ") + type;
}

/* 
 * =====================
 * LogregCostLayer
 * =====================
 */
LogregCostLayer::LogregCostLayer(ConvNet* convNet, PyObject* paramsDict) :
		CostLayer(convNet, paramsDict, false) {
}

void LogregCostLayer::fpropActs(int inpIdx, float scaleTargets,
		PASS_TYPE passType) {
	// This layer uses its two inputs together
	if (inpIdx == 0) {
		NVMatrix& labels = *_inputs[0];
		NVMatrix& probs = *_inputs[1];
		int numCases = labels.getNumElements();

		//int numOut=probs.getNumRows();
		//assert(numCases==probs.getNumCols());
		//printf("numCases:%d numOut:%d\n",numCases,numOut);

		//Matrix *hLabels=new Matrix();
		//labels.copyToHost(*hLabels,true);
		//float maxLabel=-1;
		//for(int i=0;i<numCases;++i){
		//	if(maxLabel<hLabels->getCell(0,i))
		//		maxLabel=hLabels->getCell(0,i);
		//}
		//if(maxLabel>=999)
		//	printf("----------Error: maxlabel>999-----------\n");

		//Matrix *hProbMat=new Matrix();

		//probs.copyToHost(*hProbMat,true);
		//float probmin=hProbMat->min();
		//float probmax=hProbMat->max();
		//if(probmin<=0)
		//	printf("probmin:%10.9f\n",probmin);
		//for(int i=0;i<numCases;++i){
		//	for(int j=0;j<numOut;++j){
		//		if(hProbMat->getCell(j,i)<0){
		//			printf("enter cs 4\n");
		//			printf("find non-positive prob: %6.5f\n (i,j)=(%d,%d)\n",
		//					hProbMat->getCell(j,i),i,j);
		//		}
		//	}
		//}

		NVMatrix& trueLabelLogProbs = getActs(), correctProbs;
		computeLogregCost(labels, probs, trueLabelLogProbs, correctProbs);
		_costv.clear();
		_costv.push_back(-trueLabelLogProbs.sum());
		_costv.push_back(numCases - correctProbs.sum());

		//printf("\nLogregCostLayer::fpropActs: logreg cost : %6.5f testError:%6.5f\n",
		//		-trueLabelLogProbs.sum(),(numCases - correctProbs.sum()));
		checkNVMatrixNan(trueLabelLogProbs, _name);
	}
}

void LogregCostLayer::bpropActs(NVMatrix& v, int inpIdx, float scaleTargets,
		PASS_TYPE passType) {
	assert(inpIdx == 1);
	NVMatrix& labels = _prev[0]->getActs();
	NVMatrix& probs = _prev[1]->getActs();
	NVMatrix& target = _prev[1]->getActsGrad();
	// Numerical stability optimization: if the layer below me is a softmax layer, let it handle
	// the entire gradient computation to avoid multiplying and dividing by a near-zero quantity.
	bool doWork = _prev[1]->getNext().size() > 1
			|| _prev[1]->getType() != "softmax";
	if (doWork) {
		computeLogregGrad(labels, probs, target, scaleTargets == 1, _coeff);
	}
}

/* 
 * =====================
 * RegCostLayer
 * =====================
 */
RegCostLayer::RegCostLayer(ConvNet* convNet, PyObject* paramsDict) :
		CostLayer(convNet, paramsDict, false) {
}

void RegCostLayer::fpropActs(int inpIdx, float scaleTargets,
		PASS_TYPE passType) {
	// This layer uses its two inputs together
	if (inpIdx == 0) {
		NVMatrix& labels = *_inputs[0];
		NVMatrix& logprobs = *_inputs[1];
//        NVMatrix& probs = *_inputs[1];
		int numCases = labels.getNumElements();
		checkNVMatrixNan(logprobs, _name);

		NVMatrix& trueLabelLogProbs = getActs(), correctProbs;
		computeLogregCost(labels, logprobs, trueLabelLogProbs, correctProbs,
				true);
		_costv.clear();
		_costv.push_back(-trueLabelLogProbs.sum());
		_costv.push_back(numCases - correctProbs.sum());

		//printf("\nLogregCostLayer::fpropActs: logreg cost : %6.5f testError:%6.5f\n",
		//		-trueLabelLogProbs.sum(),(numCases - correctProbs.sum()));
		checkNVMatrixNan(trueLabelLogProbs, _name);

	}
}

void RegCostLayer::bpropActs(NVMatrix& v, int inpIdx, float scaleTargets,
		PASS_TYPE passType) {
	assert(inpIdx == 1);
	NVMatrix& labels = _prev[0]->getActs();
	NVMatrix& probs = _prev[1]->getActs();
	NVMatrix& target = _prev[1]->getActsGrad();
	// Numerical stability optimization: if the layer below me is a softmax layer, let it handle
	// the entire gradient computation to avoid multiplying and dividing by a near-zero quantity.
	bool doWork = _prev[1]->getNext().size() > 1
			|| _prev[1]->getType() != "logsoftmax";
	if (doWork) {
		printf("RegCostLayer::bpropActs doWork\n");
		// to do
		computeLogregGrad(labels, probs, target, scaleTargets == 1, _coeff);
	}
}

/*
 * =====================
 * SumOfSquaresCostLayer
 * =====================
 */
SumOfSquaresCostLayer::SumOfSquaresCostLayer(ConvNet* convNet,
		PyObject* paramsDict) :
		CostLayer(convNet, paramsDict, false) {
}

void SumOfSquaresCostLayer::fpropActs(int inpIdx, float scaleTargets,
		PASS_TYPE passType) {
	int numCases = _inputs[0]->getNumCols();
	_inputs[0]->apply(NVMatrixOps::Square(), getActs());
	_costv.clear();
// will be divided by numCases in Python
	_costv.push_back(0.5 * getActs().sum());
//	_costv.push_back(0.5 * getActs().sum() / numCases);
}

void SumOfSquaresCostLayer::bpropActs(NVMatrix& v, int inpIdx,
		float scaleTargets, PASS_TYPE passType) {
	_prev[inpIdx]->getActsGrad().add(*_inputs[0], scaleTargets, -_coeff);
}

SumOfSquaresOfDiffCostLayer::SumOfSquaresOfDiffCostLayer(ConvNet* convNet,
		PyObject* paramsDict) :
		CostLayer(convNet, paramsDict, false) {
	_relativeDiff = pyDictGetInt(paramsDict, "relativeDiff");
	printf("SumOfSquaresOfDiffCostLayer %s init. _relativeDiff : %d\n",
			_name.c_str(), _relativeDiff);
}

void SumOfSquaresOfDiffCostLayer::fpropActs(int inpIdx, float scaleTargets,
		PASS_TYPE passType) {
	// This layer uses two inputs
	if (inpIdx == 0) {
		NVMatrix& labels = *_inputs[0]; // dim * numImg. _trans=0
		NVMatrix& predLabels = *_inputs[1]; // dim * numImg. _trans=0
		predLabels.subtract(labels, getActs());

		printf("SumOfSquaresOfDiffCostLayer::fpropActs\n");
		labels.print(labels.getNumRows(), 2);
		printf("\n");
		predLabels.print(predLabels.getNumRows(), 2);
		printf("\n");
		getActs().print(getActs().getNumRows(), 2);
		printf("\n");
		exit(0);

		if (_relativeDiff) {
			getActs().eltwiseDivide(labels);
		}
//		printf("SumOfSquaresOfDiffCostLayer::fpropActs print labels\n");
//		labels.print(labels.getFollowingDim(),10);
//		printf("SumOfSquaresOfDiffCostLayer::fpropActs print predLabels\n");
//		predLabels.print(predLabels.getFollowingDim(),10);
//		printf("SumOfSquaresOfDiffCostLayer::fpropActs print getActs()\n");
//		getActs().print(labels.getFollowingDim(),10);

		int numCases = labels.getLeadingDim();
		_costv.clear();
		_costv.push_back(0.5 * getActs().norm2());
//		printf("averaged cost:%f\n",0.5 * getActs().norm2()/numCases);
	}
}

void SumOfSquaresOfDiffCostLayer::bpropActs(NVMatrix& v, int inpIdx,
		float scaleTargets, PASS_TYPE passType) {
	assert(inpIdx == 1);
	NVMatrix& labels = *_inputs[0]; // dim * numImg. _trans=0
	NVMatrix& predLabels = *_inputs[1]; // dim * numImg. _trans=0
	NVMatrix diffLabels;
	//printf("SumOfSquaresOfDiffCostLayer::bpropActs\n");
//	printf("print labels\n");
//	labels.print(labels.getFollowingDim(),labels.getLeadingDim());
//	printf("print predLabels\n");
//	predLabels.print(predLabels.getFollowingDim(),predLabels.getLeadingDim());
//	exit(0);
	//printf("scaleTargets: %f\n",scaleTargets);
	if (!_relativeDiff)
		_prev[inpIdx]->getActsGrad().add(getActs(), scaleTargets, -_coeff);
	else {
		NVMatrix& absDiff = getActs().copy();
		absDiff.eltwiseDivide(labels);
		_prev[inpIdx]->getActsGrad().add(absDiff, scaleTargets, -_coeff);
	}
}

LogSumOfSquaresOfDiffCostLayer::LogSumOfSquaresOfDiffCostLayer(ConvNet* convNet,
		PyObject* paramsDict) :
		CostLayer(convNet, paramsDict, false) {
	_scale = pyDictGetFloat(paramsDict, "scale");
	printf("layer :%s. coeff:%f scale:%f\n", _name.c_str(), _coeff, _scale);
}

// to be verified
void LogSumOfSquaresOfDiffCostLayer::fpropActs(int inpIdx, float scaleTargets,
		PASS_TYPE passType) {
	// This layer uses two inputs
	if (inpIdx == 0) {
		NVMatrix& labels = *_inputs[0]; // dim * numImg. _trans=0
		NVMatrix& predLabels = *_inputs[1]; // dim * numImg. _trans=0
		int numCases = labels.getLeadingDim();
		NVMatrix squaredDiff;
		labels.applyBinary(NVMatrixBinaryOps::SquaredDiff(), predLabels,
				squaredDiff);

		squaredDiff.sum(0, getActs());
		getActs().apply(NVMatrixOps::MultByScalar(0.5));
		printf("LogSumOfSquaresOfDiffCostLayer size: %d %d\n",
				getActs().getLeadingDim(), getActs().getFollowingDim());

		NVMatrix costs;
		getActs().apply(NVMatrixOps::MultByScalar(_scale), costs);
		costs.apply(NVMatrixOps::Logistic());
		costs.apply(NVMatrixOps::AddScalar(-0.5));

		_costv.clear();
		_costv.push_back(costs.sum() / numCases);
	}
}

void LogSumOfSquaresOfDiffCostLayer::bpropActs(NVMatrix& v, int inpIdx,
		float scaleTargets, PASS_TYPE passType) {
	assert(inpIdx == 1);
	NVMatrix& labels = *_inputs[0]; // dim * numImg. _trans=0
	NVMatrix& predLabels = *_inputs[1]; // dim * numImg. _trans=0
	NVMatrix diffLabels;
	predLabels.subtract(labels, diffLabels);

	NVMatrix a, b;
	getActs().apply(NVMatrixOps::WeightedAddScalar(_scale, 0.5 * _scale), a);
	getActs().apply(NVMatrixOps::WeightedAddScalar(-1, 0.5), b);
	a.applyBinary(NVMatrixBinaryOps::Multiply(), b);

	diffLabels.eltwiseMultByVector(a);

	printf("print out diffLabels\n");
	diffLabels.print(diffLabels.getFollowingDim(), diffLabels.getLeadingDim());
	_prev[inpIdx]->getActsGrad().add(diffLabels, scaleTargets, -_coeff);
}

CroppingLayer::CroppingLayer(ConvNet* convNet, PyObject* paramsDict) :
		Layer(convNet, paramsDict, false) {
	_channels = pyDictGetInt(paramsDict, "channels");
	_start = pyDictGetInt(paramsDict, "start");
	_end = pyDictGetInt(paramsDict, "end");
}

void CroppingLayer::fpropActs(int inpIdx, float scaleTargets,
		PASS_TYPE passType) {
//	printf("CroppingLayer %s: fpropActs\n",_name.c_str());
	cropping(*_inputs[0], getActs(), _channels, _start, _end);
}

void CroppingLayer::bpropActs(NVMatrix& v, int inpIdx, float scaleTargets,
		PASS_TYPE passType) {
//	printf("CroppingLayer %s: bpropActs\n",_name.c_str());
	croppingUndo(_prev[0]->getActs(), v, _prev[0]->getActsGrad(), _start, _end,
			scaleTargets, 1);
}

ColorEnhanceCostLayer::ColorEnhanceCostLayer(ConvNet* convNet,
		PyObject* paramsDict) :
		CostLayer(convNet, paramsDict, false) {

}

/*
 * input 0: prediction of 10/30 D transform for L-channel.   (ch*10,n)
 * input 1: original 10D pixel basis					 (segment_random_sample_num*10,n)
 * input 2: groundtruth color							 (segment_random_sample_num*ch,n)
 * */

void ColorEnhanceCostLayer::fpropActs(int inpIdx, float scaleTargets,
		PASS_TYPE passType) {
	// this layer uses 3 inputs
	if (inpIdx == 0) {
		NVMatrix& predMapping = *_inputs[0]; //shape: (ch*10,num_imgs)
		//NVMatrix& gtMapping = *_inputs[1]; // shape: (ch*10, num_imgs)
		NVMatrix& colorBasis = *_inputs[1]; // shape: (segment_random_sample_num*10,num_imgs)
		NVMatrix& gtColor = *_inputs[2]; // shape: (segment_random_sample_num*ch,num_imgs)

		int basis_dim = 10;
		int ch = predMapping.getNumRows() / basis_dim;
		int segment_random_sample_num = gtColor.getNumRows() / ch;
		if (verbose)
			printf("basis_dim %d segment_random_sample_num %d ch:%d\n",
					basis_dim, segment_random_sample_num, ch);

		//assert(predMapping.getNumCols()==gtMapping.getNumCols());
		//assert(predMapping.getNumRows() == gtMapping.getNumRows());
		assert(colorBasis.getNumCols()==predMapping.getNumCols());
		assert(gtColor.getNumCols()==predMapping.getNumCols());

		assert(basis_dim*segment_random_sample_num==colorBasis.getNumRows());

		int costOption = 1;

		if (costOption == 0) {
			// measure difference between predicted mapping and estimated mapping from segment
			// TO DO: update code
			//NVMatrix diffMapping;
			//predMapping.subtract(gtMapping, diffMapping);
			//diffMapping.eltwiseMult(colorBasis);
			//diffMapping.sum(0, getActs());
			//NVMatrix dotProd, predColor, diffColor;
			//predMapping.eltwiseMult(colorBasis, dotProd);
			//dotProd.sum(0, predColor);
			//predColor.subtract(gtColor, diffColor);

			//_costv.clear();
			//_costv.push_back(0.5 * getActs().norm2());
			//_costv.push_back(0.5 * diffColor.norm2());
		} else {
			NVMatrix *dotProd = new NVMatrix();
			NVMatrix *predColor = new NVMatrix(gtColor);
			predColor->setZero();
			for (int i = 0; i < segment_random_sample_num; ++i) {
				NVMatrix& l_color_basis = colorBasis.sliceRows(i * basis_dim,
						(i + 1) * basis_dim);
				for (int j = 0; j < ch; ++j) {
					NVMatrix& l_pred_color = predColor->sliceRows(ch * i + j,
							ch * i + j + 1);
					NVMatrix& l_pred_mapping = predMapping.sliceRows(
							j * basis_dim, (j + 1) * basis_dim);
					l_pred_mapping.eltwiseMult(l_color_basis, *dotProd);
					dotProd->sum(0, l_pred_color);
					delete &l_pred_color;
					delete &l_pred_mapping;
				}
				delete &l_color_basis;
			}
			predColor->subtract(gtColor, getActs());
			if (verbose)
				printf(
						"ColorEnhanceCostLayer::fpropActs gtColor rows:%d cols:%d\n",
						predColor->getNumRows(), predColor->getNumCols());
			_costv.clear();
			_costv.push_back(
					0.5 * getActs().norm2() / segment_random_sample_num);
			delete predColor;
			delete dotProd;
		}

	}
}

void ColorEnhanceCostLayer::bpropActs(NVMatrix& v, int inpIdx,
		float scaleTargets, PASS_TYPE passType) {
	assert(inpIdx == 0);
	NVMatrix& predMapping = *_inputs[0]; //shape: (30,num_imgs)
	//NVMatrix& gtMapping = *_inputs[1]; // shape: (30, num_imgs)
	NVMatrix& colorBasis = *_inputs[1]; // shape: (segment_random_sample_num*10,num_imgs)
	NVMatrix& gtColor = *_inputs[2]; // shape: (segment_random_sample_num*3,num_imgs)

//	NVMatrix& predMapping = *_inputs[0];
//	NVMatrix& gtMapping = *_inputs[1];
//	NVMatrix& colorBasis = *_inputs[2];
//	NVMatrix& gtColor = *_inputs[3]; // shape: (segment_random_sample_num,num_imgs)

	int basis_dim = 10;
	int ch = predMapping.getNumRows() / basis_dim;
	int segment_random_sample_num = gtColor.getNumRows() / ch;
	int num_imgs = predMapping.getNumCols();

	NVMatrix *grad1 = new NVMatrix(basis_dim * ch, num_imgs, false);
	grad1->setZero();
	NVMatrix *l_grad = new NVMatrix();
	for (int i = 0; i < segment_random_sample_num; ++i) {
		NVMatrix& l_color_basis = colorBasis.sliceRows(i * basis_dim,
				(i + 1) * basis_dim);
		for (int j = 0; j < ch; ++j) {
			NVMatrix& l_grad1 = grad1->sliceRows(j * basis_dim,
					(j + 1) * basis_dim);
			NVMatrix& l_color_diff = getActs().sliceRows(ch * i + j,
					ch * i + j + 1);
			l_color_basis.eltwiseMultByVector(l_color_diff, *l_grad);
			l_grad1.add(*l_grad, 1);
			delete &l_grad1;
			delete &l_color_diff;
		}
		if (verbose)
			printf("l_grad rows:%d cols:%d l_grad min,max:%f %f\n",
					l_grad->getNumRows(), l_grad->getNumCols(), l_grad->min(),
					l_grad->max());
		delete &l_color_basis;
	}
	float scale = 1.0f / (float) segment_random_sample_num;
	grad1->scale(scale);
	if (verbose) {
		printf("grad1 rows,cols:%d %d min,max:%f %f\n", grad1->getNumRows(),
				grad1->getNumCols(), grad1->min(), grad1->max());
		printf("basis_dim %d segment_random_sample_num:%d scale:%f\n",
				basis_dim, segment_random_sample_num, scale);
	}
	_prev[inpIdx]->getActsGrad().add(*grad1, scaleTargets, -_coeff);
	delete l_grad;
	delete grad1;
}

ColorEnhanceSeparateCostLayer::ColorEnhanceSeparateCostLayer(ConvNet* convNet,
		PyObject* paramsDict) :
		CostLayer(convNet, paramsDict, false) {

}

/*
 * input 0: predicted 10D transform for L channel.   (10,n)
 * input 1: predicted 10D transform for a channel.	  (10,n)
 * input 2: predicted 10D transform for b channel.   (10,n)
 * input 3: groundtruth of 10D transform.				 (3*10,n)
 * input 4: original 10D pixel basis					 (segment_random_sample_num*10,n)
 * input 5: groundtruth color							 (segment_random_sample_num*3,n)
 * */

void ColorEnhanceSeparateCostLayer::fpropActs(int inpIdx, float scaleTargets,
		PASS_TYPE passType) {
	// this layer uses 6 inputs
	if (inpIdx == 0) {
		NVMatrix& predMapping_L = *_inputs[0]; //shape: (10,num_imgs)
		NVMatrix& predMapping_a = *_inputs[1]; //shape: (10,num_imgs)
		NVMatrix& predMapping_b = *_inputs[2]; //shape: (10,num_imgs)
		NVMatrix& gtMapping = *_inputs[3]; // shape: (3*10, num_imgs)
		NVMatrix& colorBasis = *_inputs[4]; // shape: (segment_random_sample_num*10,num_imgs)
		NVMatrix& gtColor = *_inputs[5]; // shape: (segment_random_sample_num*3,num_imgs)

		assert(predMapping_L.getNumRows()==predMapping_a.getNumRows());
		assert(predMapping_L.getNumRows()==predMapping_b.getNumRows());
		assert(predMapping_L.getNumCols()==predMapping_a.getNumCols());
		assert(predMapping_L.getNumCols()==predMapping_b.getNumCols());
		assert(predMapping_L.getNumCols()==gtMapping.getNumCols());
		assert(predMapping_L.getNumCols()==colorBasis.getNumCols());
		assert(predMapping_L.getNumCols()==gtColor.getNumCols());
		assert(gtMapping.getNumRows()==30);
		assert(gtColor.getNumRows()%3==0);

		int basis_dim = 10;
		int segment_random_sample_num = colorBasis.getNumRows() / basis_dim;
		assert(gtColor.getNumRows()==(segment_random_sample_num*3));

		if (verbose)
			printf("basis_dim %d segment_random_sample_num %d\n", basis_dim,
					segment_random_sample_num);

		NVMatrix dotProd;
		NVMatrix predColor(gtColor);
		for (int i = 0; i < segment_random_sample_num; ++i) {
			NVMatrix& l_color_basis = colorBasis.sliceRows(i * basis_dim,
					(i + 1) * basis_dim);
			NVMatrix& l_pred_L = predColor.sliceRows(3 * i, 3 * i + 1);
			NVMatrix& l_pred_a = predColor.sliceRows(3 * i + 1, 3 * i + 2);
			NVMatrix& l_pred_b = predColor.sliceRows(3 * i + 2, 3 * i + 3);
			predMapping_L.eltwiseMult(l_color_basis, dotProd);
			dotProd.sum(0, l_pred_L);
			predMapping_a.eltwiseMult(l_color_basis, dotProd);
			dotProd.sum(0, l_pred_a);
			predMapping_b.eltwiseMult(l_color_basis, dotProd);
			dotProd.sum(0, l_pred_b);
			delete &l_color_basis;
			delete &l_pred_L;
			delete &l_pred_a;
			delete &l_pred_b;
		}
		predColor.subtract(gtColor, getActs());
		_costv.clear();
		_costv.push_back(0.5 * getActs().norm2() / segment_random_sample_num);
	}

}

void ColorEnhanceSeparateCostLayer::bpropActs(NVMatrix& v, int inpIdx,
		float scaleTargets, PASS_TYPE passType) {
	assert(inpIdx == 0 || inpIdx == 1 || inpIdx ==2);
//	printf("inpIdx: %d\n",inpIdx);
	NVMatrix& predMapping = *_inputs[inpIdx]; //shape: (10,num_imgs)
	NVMatrix& gtMapping = *_inputs[3]; // shape: (3*10, num_imgs)
	NVMatrix& colorBasis = *_inputs[4]; // shape: (segment_random_sample_num*10,num_imgs)
	NVMatrix& gtColor = *_inputs[5]; // shape: (segment_random_sample_num*3,num_imgs)

	int basis_dim = 10;
	int segment_random_sample_num = colorBasis.getNumRows() / basis_dim;
	assert(gtColor.getNumRows()==(segment_random_sample_num*3));
	int num_imgs = predMapping.getNumCols();

	NVMatrix grad1, grad2;
	grad1.resize(basis_dim, num_imgs);
	grad1.setZero();
	grad2.resize(basis_dim, num_imgs);
	for (int i = 0; i < segment_random_sample_num; ++i) {
		NVMatrix l_grad;
		NVMatrix& l_color_basis = colorBasis.sliceRows(i * basis_dim,
				(i + 1) * basis_dim);
		NVMatrix& l_color_diff = getActs().sliceRows(3 * i + inpIdx,
				3 * i + inpIdx + 1);
		l_color_basis.eltwiseMultByVector(l_color_diff, l_grad);
		l_grad.add(grad1, 1, 1, grad2);
		grad2.copy(grad1);
		delete &l_color_basis;
		delete &l_color_diff;
	}
	grad1.scale(1.0 / (float) segment_random_sample_num);
	_prev[inpIdx]->getActsGrad().add(grad1, scaleTargets, -_coeff);

}

ConcatenateLayer::ConcatenateLayer(ConvNet* convNet, PyObject* paramsDict) :
		Layer(convNet, paramsDict, false) {

}

void ConcatenateLayer::fpropPreCommon(NVMatrixV& v, PASS_TYPE passType) {
	assert(v.size()>0);
	int outputs = 0;
	int num_imgs = v[0]->getNumCols();
	for (int i = 0; i < v.size(); ++i) {
		assert(num_imgs==v[i]->getNumCols());
		outputs += v[i]->getNumRows();
	}
	if (verbose)
		printf("ConcatenateLayer::fpropPreCommon outputs:%d\n", outputs);
	getActs().resize(outputs, num_imgs);
	_outputs_p = 0;
}

void ConcatenateLayer::fpropActs(int inpIdx, float scaleTargets,
		PASS_TYPE passType) {
	int input_dim = _inputs[inpIdx]->getNumRows();
//	printf("ConcatenateLayer::fpropActs inpIdx:%d input_dim:%d\n",
//			inpIdx,input_dim);
	NVMatrix& dest = getActs().sliceRows(_outputs_p, _outputs_p + input_dim);
	_inputs[inpIdx]->copy(dest);
	_outputs_p += input_dim;
	delete &dest;
}

void ConcatenateLayer::bpropPreCommon(NVMatrix& v, PASS_TYPE passType) {
	_outputs_p = 0;
}

void ConcatenateLayer::bpropActs(NVMatrix& v, int inpIdx, float scaleTargets,
		PASS_TYPE passType) {
	int input_dim = _inputs[inpIdx]->getNumRows();
//	printf("ConcatenateLayer::bpropActs inpIdx:%d input_dim:%d\n",
//			inpIdx,input_dim);
	NVMatrix& l_grad = v.sliceRows(_outputs_p, _outputs_p + input_dim);
	l_grad.copy(_prev[inpIdx]->getActsGrad());
	_outputs_p += input_dim;
	delete &l_grad;
}

GradMagEnhanceCostLayer::GradMagEnhanceCostLayer(ConvNet* convNet,
		PyObject* paramsDict) :
		CostLayer(convNet, paramsDict, false) {

}

/*
 * input 0: prediction of 2 D transform for gradient magnitude.   (2,n)
 * input 1: pixel L channel in input image				(1,n)
 * input 2: pixel L channel gradient magnitude in input image				(1,n)
 * input 3: pixel L channel gradient magnitude in enhanced image				(1,n)
 * */
void GradMagEnhanceCostLayer::fpropActs(int inpIdx, float scaleTargets,
		PASS_TYPE passType) {
	if (inpIdx == 0) {
		NVMatrix& predMapping = *_inputs[0]; //shape: (2,num_imgs)
		NVMatrix& inL = *_inputs[1]; // shape: (1,num_imgs)
		NVMatrix& inGradMag = *_inputs[2]; // shape: (1,num_imgs)
		NVMatrix& enhGradMag = *_inputs[3]; // shape: (1,num_imgs)

		assert(predMapping.getNumRows()==2);
		assert(inL.getNumRows()==1);
		assert(inGradMag.getNumRows()==1);
		assert(enhGradMag.getNumRows()==1);

//		printf("GradMagEnhanceCostLayer::fpropActs\n");
// log(enhGradMag/inGradMag)=a*logInL+b
// cost function: 0.5 * (squared difference in enhGradMag)
//		printf("inL min:%f \n",inL.min());

		inL.apply(NVMatrixOps::Log(), logInL);
		NVMatrix predLogRatio;
		NVMatrix& coeff_a = predMapping.sliceRows(0, 1);
		NVMatrix& coeff_b = predMapping.sliceRows(1, 2);
//		printf("coeff_a mean %f \n", coeff_a.mean());
//		printf("coeff_b mean %f \n", coeff_b.mean());
		logInL.eltwiseMult(coeff_a, predLogRatio);
		predLogRatio.applyBinary(NVMatrixBinaryOps::Add(), coeff_b);
		predLogRatio.apply(NVMatrixOps::Exp(), predEnhGradMag);
//		printf("inGradMag min:%f max:%f\n",inGradMag.min(),inGradMag.max());
		predEnhGradMag.applyBinary(NVMatrixBinaryOps::Multiply(), inGradMag);
//		printf("enhGradMag min:%f max:%f\n",enhGradMag.min(),enhGradMag.max());
		predEnhGradMag.subtract(enhGradMag, getActs());

		_costv.clear();
		_costv.push_back(0.5 * getActs().norm2());

		delete &coeff_a;
		delete &coeff_b;
//		printf("exit GradMagEnhanceCostLayer::fpropActs\n");
	}
}

void GradMagEnhanceCostLayer::bpropActs(NVMatrix& v, int inpIdx,
		float scaleTargets, PASS_TYPE passType) {
//	printf("GradMagEnhanceCostLayer::bpropActs tag1\n");
	assert(inpIdx==0);
	NVMatrix& predMapping = *_inputs[0]; //shape: (2,num_imgs)
	NVMatrix& inL = *_inputs[1]; // shape: (1,num_imgs)
	NVMatrix& inGradMag = *_inputs[2]; // shape: (1,num_imgs)
	NVMatrix& enhGradMag = *_inputs[3]; // shape: (1,num_imgs)

	int num_imgs=predMapping.getNumCols();

	//if(_prev[inpIdx]->getActsGrad().getNumRows()==0){
		_prev[inpIdx]->getActsGrad().resize(2,num_imgs);
		_prev[inpIdx]->getActsGrad().setZero();
	//}
	NVMatrix& prev_grad_coeff_a = _prev[inpIdx]->getActsGrad().sliceRows(0,1);
	NVMatrix& prev_grad_coeff_b = _prev[inpIdx]->getActsGrad().sliceRows(1,2);
//	printf("GradMagEnhanceCostLayer::bpropActs tag2\n");

	NVMatrix grad_coeff_a,grad_coeff_b;
	getActs().eltwiseMult(predEnhGradMag,grad_coeff_b);
	grad_coeff_b.eltwiseMult(logInL,grad_coeff_a);
//
//	printf("grad_coeff_a rows,cols:%d %d\n",grad_coeff_a.getNumRows(),
//			grad_coeff_a.getNumCols());
//	printf("grad_coeff_b rows,cols:%d %d\n",grad_coeff_b.getNumRows(),
//			grad_coeff_b.getNumCols());

	prev_grad_coeff_a.add(grad_coeff_a,scaleTargets,-_coeff);
	prev_grad_coeff_b.add(grad_coeff_b,scaleTargets,-_coeff);

	delete &prev_grad_coeff_a;
	delete &prev_grad_coeff_b;
//	printf("GradMagEnhanceCostLayer::bpropActs tag3\n");
}

