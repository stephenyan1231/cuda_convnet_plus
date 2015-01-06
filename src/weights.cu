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

#include <weights.cuh>

bool Weights::_autoCopyToGPU = false;

extern bool verbose;
// Scale your gradient by epsW / numCases!
void Weights::update() {
	// Only true owner of weights updates
	if (_srcWeights == NULL && (_epsW * _epsW_scale) > 0) {
		assert(_onGPU);
		if (_useGrad) {
			if (verbose) {
//				printf("Weights::update _weightsInc rows,cols:%d %d sum:%f\n",
//				_weightsInc->getNumRows(),_weightsInc->getNumCols(),_weightsInc->sum());
			}
			_weightsInc->add(*_weightsGrad, _mom, 1);
		}
		if (_wc > 0) {
			if (verbose) {
				//printf("Weights::update _weightsInc rows,cols:%d %d sum:%f\n",
				//_weightsInc->getNumRows(),_weightsInc->getNumCols(),_weightsInc->sum());
			}
			_weightsInc->add(*_weights, -_wc * _epsW * _epsW_scale);
		}
		if (verbose) {
			NVMatrix tmp;
			_weights->apply(NVMatrixOps::Abs(), tmp);
			float meanAbsWeights = tmp.sum() / tmp.getNumElements();
			_weightsInc->apply(NVMatrixOps::Abs(), tmp);
			float meanAbsWeightsInc = tmp.sum() / tmp.getNumElements();
			printf(
					"Weights::update _weightsInc rows,cols:%d %d meanAbsWeights:%f meanAbsWeightsInc:%10.9f\n",
					_weightsInc->getNumRows(), _weightsInc->getNumCols(),
					meanAbsWeights, meanAbsWeightsInc);
			/*printf("_weights\n");
			_weights->print(1,_weights->getNumCols());
			printf("_weightsInc\n");
			_weightsInc->print(1,_weightsInc->getNumCols());*/
		}
		_weights->add(*_weightsInc);
		_numUpdates = 0;
	}
}
