Copyright (c) 2015 The University of Edinburgh.
 
This software was developed as part of the                       
EC FP7 funded project Adept (Project ID: 610490)                 
    www.adept-project.eu                                            

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

# Adept Micro Benchmarks - SIMD

This README file introduces the SIMD Micro Benchmarks.

## Citation & Further Information
If you would like to cite this work, please cite:
Nick Johnson et al., "Adept Deliverable D2.3 - Updated Report on Adept Benchmarks", September 2015.
available at http://www.adept-project.eu/images/Deliverables/Adept%20D2.3.pdf


## Basic Operations

A benchmark to measure the basic operations (add, sub, mult, div) on scalar data types (int, long, float, double).

Each run of the benchmark consists of:

- a warm-up loop
- a timed loop with a `nop` instruction
- an empty timed loop
- loops with the basic operation (using SIMD instructions) of choice, with increasing work per iteration and correspondingly reduced number of iterations (i.e. the number of iteration remains consistent).
  
The `nop` and empty loops are used to get an understanding of the overheads incurred by the loop. Similary, increasing the number of operations per loop iterations while reducing the number of iterations gives an idea of how work inside a loop impacts performance and power.

The user can choose if the data used as part of the operations is volatile (i.e. read from memory at each access). The options are no volatile variables (default), ONEVOL where 1 of the variable is declared volatile, or ALLVOL where all the variables are volatile.