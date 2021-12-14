#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <float.h>
#include "uwnet.h"


// Run a maxpool layer on input
// layer l: pointer to layer to run
// matrix in: input to layer
// returns: the result of running the layer
matrix forward_maxpool_layer(layer l, matrix in)
{
    // Saving our input
    // Probably don't change this
    free_matrix(*l.x);
    *l.x = copy_matrix(in);

    int outw = (l.width-1)/l.stride + 1;
    int outh = (l.height-1)/l.stride + 1;
    matrix out = make_matrix(in.rows, outw*outh*l.channels);

    // TODO: 6.1 - iterate over the input and fill in the output with max values
    int cols = l.width*l.height;
    //printf("forward in: %i, %i\n", in.cols, in.rows);
    
    for (int batch = 0; batch < in.rows; batch++) {
        for (int c = 0; c < l.channels; c++) {
            for (int i = 0; i < outh; i++) {
                for (int j = 0; j < outw; j++) {
                    float maximum = -INFINITY;
                    for (int kernel_y = 0; kernel_y < l.size; kernel_y ++) {
                        for (int kernel_x = 0; kernel_x < l.size; kernel_x ++) {
                            // calculate the relative corrdinates
                            int x_c = j*l.stride -(l.size - 1) / 2 + kernel_x;
                            int y_c = i*l.stride -(l.size - 1) / 2 + kernel_y;
                            long channeloffset = c*cols;
                            if (x_c >= 0 && y_c >= 0 && x_c < l.width && y_c < l.height){
                                if (in.data[batch*in.cols + channeloffset + y_c * l.width + x_c] > maximum) {
                                    maximum = in.data[batch*in.cols + channeloffset + y_c * l.width + x_c];
                                }
                            }
                        }
                    }
                    out.data[batch*out.cols + j + i * outw + c*outw*outh] = maximum;
                }
            }
        }
    }
    return out;
}

// Run a maxpool layer backward
// layer l: layer to run
// matrix dy: error term for the previous layer
matrix backward_maxpool_layer(layer l, matrix dy)
{
    matrix in    = *l.x;
    matrix dx = make_matrix(dy.rows, l.width*l.height*l.channels);

    int outw = (l.width-1)/l.stride + 1;
    int outh = (l.height-1)/l.stride + 1;
    // TODO: 6.2 - find the max values in the input again and fill in the
    // corresponding delta with the delta from the output. This should be
    // similar to the forward method in structure.
    int cols = l.width*l.height;
    
    for (int batch = 0; batch < in.rows; batch++) {
        for (int c = 0; c < l.channels; c++) {
            for (int i = 0; i < outh; i++) {
                for (int j = 0; j < outw; j++) {
                    float maximum = -INFINITY;
                    int max_index_j = 0;
                    int max_index_i = 0;
                    for (int kernel_y = 0; kernel_y < l.size; kernel_y ++) {
                        for (int kernel_x = 0; kernel_x < l.size; kernel_x ++) {
                            // calculate the relative corrdinates
                            int x_c = j*l.stride - (l.size - 1) / 2 + kernel_x;
                            int y_c = i*l.stride - (l.size - 1) / 2 + kernel_y;
                            long channeloffset = c*cols;
                            if (x_c >= 0 && y_c >= 0 && x_c < l.width && y_c < l.height){
                                if (in.data[batch*in.cols + channeloffset + y_c * l.width + x_c] > maximum) {
                                    maximum = in.data[batch*in.cols + channeloffset + y_c * l.width + x_c];
                                    max_index_j = x_c;
                                    max_index_i = y_c;
                                }
                            }
                        }
                    }
                    dx.data[batch*dx.cols + max_index_j + max_index_i*l.width + c*l.width*l.height] 
                            += dy.data[batch*dy.cols + j + i*outw + c*outw*outh];  
                }
            }
        }
    }
    return dx;
}

// Update maxpool layer
// Leave this blank since maxpool layers have no update
void update_maxpool_layer(layer l, float rate, float momentum, float decay){}

// Make a new maxpool layer
// int w: width of input image
// int h: height of input image
// int c: number of channels
// int size: size of maxpool filter to apply
// int stride: stride of operation
layer make_maxpool_layer(int w, int h, int c, int size, int stride)
{
    layer l = {0};
    l.width = w;
    l.height = h;
    l.channels = c;
    l.size = size;
    l.stride = stride;
    l.x = calloc(1, sizeof(matrix));
    l.forward  = forward_maxpool_layer;
    l.backward = backward_maxpool_layer;
    l.update   = update_maxpool_layer;
    return l;
}

