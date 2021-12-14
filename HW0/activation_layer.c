#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include "uwnet.h"


// Run an activation layer on input
// layer l: pointer to layer to run
// matrix x: input to layer
// returns: the result of running the layer y = f(x)
matrix forward_activation_layer(layer l, matrix x)
{
    // Saving our input
    // Probably don't change this
    free_matrix(*l.x);
    *l.x = copy_matrix(x);

    ACTIVATION a = l.activation;
    matrix y = copy_matrix(x);

    // TODO: 2.1
    // apply the activation function to matrix y
    // logistic(x) = 1/(1+e^(-x))
    // relu(x)     = x if x > 0 else 0
    // lrelu(x)    = x if x > 0 else .01 * x
    for (int i = 0; i < y.rows; i++) {
        float sum = 0;
        for (int j = 0; j < y.cols; j++) {
        	if (a == LINEAR) {
        		;
        	}
        	else if (a == LOGISTIC) {
        		y.data[j+i*y.cols] = 1.0/(1.0+exp(-x.data[j+i*x.cols]));
        	}
        	else if (a == RELU) {
        		y.data[j+i*y.cols] = (x.data[j+i*x.cols] > 0.0) ? x.data[j+i*x.cols]:0.0;
        	}
        	else if (a == LRELU) {
        		y.data[j+i*y.cols] = (x.data[j+i*x.cols] > 0.0) ? x.data[j+i*x.cols]:0.01*x.data[j+i*x.cols];
        	}
        	else if (a == SOFTMAX) {
        		y.data[j+i*y.cols] = exp(x.data[j+i*x.cols]);
        	}
        	sum += y.data[j+i*y.cols];
        }
        if (a == SOFTMAX) {
        	for (int j = 0; j < y.cols; j++) {
        		y.data[j+i*y.cols] /= sum;
        	}
        }
    }
    return y;
}

// Run an activation layer on input
// layer l: pointer to layer to run
// matrix dy: derivative of loss wrt output, dL/dy
// returns: derivative of loss wrt input, dL/dx
matrix backward_activation_layer(layer l, matrix dy)
{
    matrix x = *l.x;
    matrix dx = copy_matrix(dy);
    ACTIVATION a = l.activation;

    // TODO: 2.2
    // calculate dL/dx = f'(x) * dL/dy
    // assume for this part that f'(x) = 1 for softmax because we will only use
    // it with cross-entropy loss for classification and include it in the loss
    // calculations
    // d/dx logistic(x) = logistic(x) * (1 - logistic(x))
    // d/dx relu(x)     = 1 if x > 0 else 0
    // d/dx lrelu(x)    = 1 if x > 0 else 0.01
    // d/dx softmax(x)  = 1
    for (int i = 0; i < x.rows; i++) {
        for (int j = 0; j < x.cols; j++) {
        	double grad = 1.0;
        	if (a == LINEAR || a == SOFTMAX) {
                ;
        	}
        	else if (a == LOGISTIC) {
                grad = 1.0/(1.0+exp(-x.data[j+i*x.cols]))*(1.0 - 1.0/(1.0+exp(-x.data[j+i*x.cols])));
        	}
        	else if (a == RELU) {
        		grad = (x.data[j+i*x.cols] > 0) ? 1.0:0.0;
        	}
        	else if (a == LRELU) {
        		grad = (x.data[j+i*x.cols] > 0) ? 1.0:0.01;
        	}
        	dx.data[j+i*dx.cols] *= grad;
        }
    }
    return dx;
}

// Update activation layer..... nothing happens tho
// layer l: layer to update
// float rate: SGD learning rate
// float momentum: SGD momentum term
// float decay: l2 normalization term
void update_activation_layer(layer l, float rate, float momentum, float decay){}

layer make_activation_layer(ACTIVATION a)
{
    layer l = {0};
    l.activation = a;
    l.x = calloc(1, sizeof(matrix));
    l.forward = forward_activation_layer;
    l.backward = backward_activation_layer;
    l.update = update_activation_layer;
    return l;
}
