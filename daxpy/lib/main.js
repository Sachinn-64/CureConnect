/**
* @license Apache-2.0
*
* Copyright (c) 2026 The Stdlib Authors.
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*    http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/

'use strict';

// MODULES //

var numelDimension = require( '@stdlib/ndarray/base/numel-dimension' );
var getStride = require( '@stdlib/ndarray/base/stride' );
var getOffset = require( '@stdlib/ndarray/base/offset' );
var getData = require( '@stdlib/ndarray/base/data-buffer' );
var strided = require( '@stdlib/blas/base/daxpy' ).ndarray;


// MAIN //

/**
* Multiplies a one-dimensional double-precision floating-point ndarray `x` by a constant `alpha` and add the result to a one-dimensional double-precision floating-point ndarray `y`.
*
* @param {number} alpha - scalar constant
* @param {ArrayLikeObject<Object>} arrays - array-like object containing an input ndarray and an output ndarray
* @returns {Object} output ndarray
*
* @example
* var Float64Array = require( '@stdlib/array/float64' );
* var ndarray = require( '@stdlib/ndarray/base/ctor' );
*
* var xbuf = new Float64Array( [ 1.0, 2.0, 3.0, 4.0, 5.0 ] );
* var x = new ndarray( 'float64', xbuf, [ 5 ], [ 1 ], 0, 'row-major' );
*
* var ybuf = new Float64Array( [ 1.0, 1.0, 1.0, 1.0, 1.0 ] );
* var y = new ndarray( 'float64', ybuf, [ 5 ], [ 1 ], 0, 'row-major' );
*
* var out = daxpy( 5.0, [ x, y ] );
* // returns <ndarray>[ 6.0, 11.0, 16.0, 21.0, 26.0 ]
*/
function daxpy( alpha, arrays ) {
	var x = arrays[ 0 ];
	var y = arrays[ 1 ];
	strided( numelDimension( x, 0 ), alpha, getData( x ), getStride( x, 0 ), getOffset( x ), getData( y ), getStride( y, 0 ), getOffset( y ) ); // eslint-disable-line max-len
	return y;
}


// EXPORTS //

module.exports = daxpy;
