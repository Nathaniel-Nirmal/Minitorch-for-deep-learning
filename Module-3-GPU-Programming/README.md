### Parallel Check Output

MAP
 
================================================================================
 Parallel Accelerator Optimizing:  Function tensor_map.<locals>._map, <br />
/Users/nathaniel/Desktop/MLE/workspace/mle- <br />
module-3-Nathaniel-Nirmal/minitorch/fast_ops.py (154) <br />  
================================================================================


Parallel loop listing for  Function tensor_map.<locals>._map, /Users/nathaniel/Desktop/MLE/workspace/mle-module-3-Nathaniel-Nirmal/minitorch/fast_ops.py (154) <br />
--------------------------------------------------------------------------------------|loop #ID<br />
    def _map(                                                                         | <br />
        out: Storage,                                                                 | <br />
        out_shape: Shape,                                                             | <br />
        out_strides: Strides,                                                         | <br />
        in_storage: Storage,                                                          | <br />
        in_shape: Shape,                                                              | <br />
        in_strides: Strides,                                                          | <br />
    ) -> None:                                                                        | <br />
        # TODO: Implement for Task 3.1.                                               | <br />
        stride_aligned = True                                                         | <br />
        if len(in_shape) != len(out_shape) or len(in_strides) != len(out_strides):    | <br />
            stride_aligned = False                                                    | <br />
        else:                                                                         | <br />
            for d in range(MAX_DIMS):                                                 | <br />
                if out_strides[d] != in_strides[d] or out_shape[d] != in_shape[d]:    | <br />
                    stride_aligned = False                                            | <br />
                    break                                                             | <br />
                                                                                      | <br />
        if stride_aligned:                                                            | <br />
            for i in prange(len(out)):------------------------------------------------| #2 <br />
                out[i] = fn(in_storage[i])                                            | <br />
                                                                                      | <br />
        else:                                                                         | <br />
            for i in prange(len(out)):------------------------------------------------| #3 <br />
                out_index = np.zeros(MAX_DIMS, np.int32)------------------------------| #0 <br />
                in_index = np.zeros(MAX_DIMS, np.int32)-------------------------------| #1 <br />
                to_index(i, out_shape, out_index)                                     | <br />
                broadcast_index(out_index, out_shape, in_shape, in_index)             | <br />
                o = index_to_position(out_index, out_strides)                         | <br />
                j = index_to_position(in_index, in_strides)                           | <br />
                out[o] = fn(in_storage[j])                                            | <br />
--------------------------------- Fusing loops ---------------------------------<br />
Attempting fusion of parallel loops (combines loops with similar properties)...<br />
 
Fused loop summary:<br />
+--0 has the following loops fused into it:<br />
   +--1 (fused)<br />
Following the attempted fusion of parallel for-loops there are 3 parallel for-<br />
loop(s) (originating from loops labelled: #2, #3, #0).<br />
--------------------------------------------------------------------------------
---------------------------- Optimising loop nests -----------------------------<br />
Attempting loop nest rewrites (optimising for the largest parallel loops)...<br />
 
+--3 is a parallel loop<br />
   +--0 --> rewritten as a serial loop<br />
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------<br />
Parallel region 0:<br />
+--3 (parallel)<br />
   +--0 (parallel)<br />
   +--1 (parallel)<br />


--------------------------------------------------------------------------------<br />
------------------------------ After Optimisation ------------------------------<br />
Parallel region 0:<br />
+--3 (parallel)<br />
   +--0 (serial, fused with loop(s): 1)<br />


 
Parallel region 0 (loop #3) had 1 loop(s) fused and 1 loop(s) serialized as part<br />
 of the larger parallel loop (#3).<br />
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
 
---------------------------Loop invariant code motion---------------------------<br />
Allocation hoisting:<br />
The memory allocation derived from the instruction at <br />
/Users/nathaniel/Desktop/MLE/workspace/mle-<br />
module-3-Nathaniel-Nirmal/minitorch/fast_ops.py (178) is hoisted out of the parallel <br />
loop labelled #3 (it will be performed before the loop is executed and reused <br />
inside the loop):<br />
   Allocation:: out_index = np.zeros(MAX_DIMS, np.int32)<br />
    - numpy.empty() is used for the allocation.<br />
The memory allocation derived from the instruction at <br />
/Users/nathaniel/Desktop/MLE/workspace/mle-<br />
module-3-Nathaniel-Nirmal/minitorch/fast_ops.py (179) is hoisted out of the parallel <br />
loop labelled #3 (it will be performed before the loop is executed and reused <br />
inside the loop):<br />
   Allocation:: in_index = np.zeros(MAX_DIMS, np.int32)<br />
    - numpy.empty() is used for the allocation.<br />
None<br />
ZIP<br />
 
================================================================================
 Parallel Accelerator Optimizing:  Function tensor_zip.<locals>._zip, <br />
/Users/nathaniel/Desktop/MLE/workspace/mle-<br />
module-3-Nathaniel-Nirmal/minitorch/fast_ops.py (212)  <br />
================================================================================


Parallel loop listing for  Function tensor_zip.<locals>._zip, /Users/nathaniel/Desktop/MLE/workspace/mle-module-3-Nathaniel-Nirmal/minitorch/fast_ops.py (212) <br />
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|loop #ID<br />
    def _zip<br />(                                                                                                                                                                                                   | <br />
        out: Storage,                                                                                                                                                                                           | <br />
        out_shape: Shape,                                                                                                                                                                                       | <br />
        out_strides: Strides,                                                                                                                                                                                   | <br />
        a_storage: Storage,                                                                                                                                                                                     | <br />
        a_shape: Shape,                                                                                                                                                                                         | <br />
        a_strides: Strides,                                                                                                                                                                                     | <br />
        b_storage: Storage,                                                                                                                                                                                     | <br />
        b_shape: Shape,                                                                                                                                                                                         | <br />
        b_strides: Strides,                                                                                                                                                                                     | <br />
    ) -> None:                                                                                                                                                                                                  | <br />
        # TODO: Implement for Task 3.1.                                                                                                                                                                         | 
                                                                                                                                                                                                                | <br />
        # copy both tensors into local space                                                                                                                                                                    | 
        # using threads, access into each local space and write to out                                                                                                                                          | 
                                                                                                                                                                                                                | <br />
        MAX_DIMS = len(out_shape)                                                                                                                                                                               | <br />
                                                                                                                                                                                                                | <br />
        stride_aligned = True                                                                                                                                                                                   | <br />
        if len(a_shape) == len(b_shape) and len(a_shape) == len(out_shape) and len(a_strides) == len(b_strides) and len(a_strides) == len(out_strides):       <br />                                                  | 
            for d in range(MAX_DIMS):                                                                                                                                                                           | 
                if out_strides[d] != a_strides[d] or out_strides[d] != b_strides[d] or b_strides[d] != a_strides[d] or out_shape[d] != a_shape[d] or out_shape[d] != b_shape[d] or b_shape[d] != a_shape[d]:    | <br />
                    stride_aligned = False                                                                                                                                                                      | <br />
                    break                                                                                                                                                                                       | <br />
        else:                                                                                                                                                                                                   | <br />
            stride_aligned = False                                                                                                                                                                              | <br />
                                                                                                                                                                                                                | <br />
        if stride_aligned:                                                                                                                                                                                      | <br />
            for i in prange(len(out)):--------------------------------------------------------------------------------------------------------------------------------------------------------------------------| #7<br />
                out[i] = fn(a_storage[i], b_storage<br />[i])                                                                                                                                                         | <br />
        else:                                                                                                                                                                                                   | <br />
            for i in prange(len(out)):--------------------------------------------------------------------------------------------------------------------------------------------------------------------------| #8<br />
                out_index = np.zeros(MAX_DIMS, np.int32)--------------------------------------------------------------------------------------------------------------------------------------------------------| #4<br />
                a_index = np.zeros(MAX_DIMS, np.int32)----------------------------------------------------------------------------------------------------------------------------------------------------------| #5<br />
                b_index = np.zeros(MAX_DIMS, np.int32)----------------------------------------------------------------------------------------------------------------------------------------------------------| #6<br />
                to_index(i, out_shape, out_index)                                                                                                                                                               | <br />
                o = index_to_position(out_index, out_strides)                                                                                                                                                   | <br />
                broadcast_index(out_index, out_shape, a_shape, a_index)                                                                                                                                         | <br />
                j = index_to_position(a_index, a_strides)                                                                                                                                                       | <br />
                broadcast_index(out_index, out_shape, b_shape, b_index)                                                                                                                                         | <br />
                k = index_to_position(b_index, b_strides)                                                                                                                                                       | <br />
                out[o] = fn(a_storage[j], b_storage[k])                                                                                                                                                         | <br />
--------------------------------- Fusing loops ---------------------------------<br />
Attempting fusion of parallel loops (combines loops with similar properties)...<br />
 
Fused loop summary:<br />
+--4 has the following loops fused into it:<br />
   +--5 (fused)<br />
   +--6 (fused)<br />
Following the attempted fusion of parallel for-loops there are 3 parallel for-<br />
loop(s) (originating from loops labelled: #7, #8, #4).<br />
--------------------------------------------------------------------------------
---------------------------- Optimising loop nests -----------------------------<br />
Attempting loop nest rewrites (optimising for the largest parallel loops)...<br />
 
+--8 is a parallel loop<br />
   +--4 --> rewritten as a serial loop<br />
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------<br />
Parallel region 0:<br />
+--8 (parallel)<br />
   +--4 (parallel)<br />
   +--5 (parallel)<br />
   +--6 (parallel)<br />


--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel region 0:<br />
+--8 (parallel)<br />
   +--4 (serial, fused with loop(s): 5, 6)<br />


 
Parallel region 0 (loop #8) had 2 loop(s) fused and 1 loop(s) serialized as part<br />
 of the larger parallel loop (#8).<br />
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
 
---------------------------Loop invariant code motion---------------------------
Allocation hoisting:<br />
The memory allocation derived from the instruction at <br />
/Users/nathaniel/Desktop/MLE/workspace/mle-<br />
module-3-Nathaniel-Nirmal/minitorch/fast_ops.py (244) is hoisted out of the parallel <br />
loop labelled #8 (it will be performed before the loop is executed and reused <br />
inside the loop):<br />
   Allocation:: out_index = np.zeros(MAX_DIMS, np.int32)<br />
    - numpy.empty() is used for the allocation.<br />
The memory allocation derived from the instruction at <br />
/Users/nathaniel/Desktop/MLE/workspace/mle-<br />
module-3-Nathaniel-Nirmal/minitorch/fast_ops.py (245) is hoisted out of the parallel <br />
loop labelled #8 (it will be performed before the loop is executed and reused <br />
inside the loop):<br />
   Allocation:: a_index = np.zeros(MAX_DIMS, np.int32)<br />
    - numpy.empty() is used for the allocation.<br />
The memory allocation derived from the instruction at <br />
/Users/nathaniel/Desktop/MLE/workspace/mle-<br />
module-3-Nathaniel-Nirmal/minitorch/fast_ops.py (246) is hoisted out of the parallel <br />
loop labelled #8 (it will be performed before the loop is executed and reused <br />
inside the loop):<br />
   Allocation:: b_index = np.zeros(MAX_DIMS, np.int32)<br />
    - numpy.empty() is used for the allocation.<br />
None<br />
REDUCE<br />
 
================================================================================
 Parallel Accelerator Optimizing:  Function tensor_reduce.<locals>._reduce, <br />
/Users/nathaniel/Desktop/MLE/workspace/mle-<br />
module-3-Nathaniel-Nirmal/minitorch/fast_ops.py (277)  <br />
================================================================================


Parallel loop listing for  Function tensor_reduce.<locals>._reduce, /Users/nathaniel/Desktop/MLE/workspace/mle-module-3-Nathaniel-Nirmal/minitorch/fast_ops.py (277) <br />
---------------------------------------------------------------|loop #ID<br />
    def _reduce(                                               | <br />
        out: Storage,                                          | <br />
        out_shape: Shape,                                      | <br />
        out_strides: Strides,                                  | <br />
        a_storage: Storage,                                    | <br />
        a_shape: Shape,                                        | <br />
        a_strides: Strides,                                    | <br />
        reduce_dim: int,                                       | <br />
    ) -> None:                                                 | <br />
        # TODO: Implement for Task 3.1.                        | <br />
                                                               | <br />
        MAX_DIMS = len(out_shape)                              | <br />
                                                               | <br />
        for i in prange(len(out)):-----------------------------| #10 <br />
            out_index: Index = np.zeros(MAX_DIMS, np.int32)----| #9 <br />
            reduce_size = a_shape[reduce_dim]                  | <br />
            to_index(i, out_shape, out_index)                  | <br />
            o = index_to_position(out_index, out_strides)      | <br />
            for s in range(reduce_size):                       | <br />
                out_index[reduce_dim] = s                      | <br />
                j = index_to_position(out_index, a_strides)    | <br />
                out[o] = fn(out[o], a_storage[j])              | <br />
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...<br />
Following the attempted fusion of parallel for-loops there are 2 parallel for-<br />
loop(s) (originating from loops labelled: #10, #9).<br />
--------------------------------------------------------------------------------
---------------------------- Optimising loop nests -----------------------------
Attempting loop nest rewrites (optimising for the largest parallel loops)...<br />
 
+--10 is a parallel loop<br />
   +--9 --> rewritten as a serial loop<br />
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------<br />
Parallel region 0:<br />
+--10 (parallel)<br />
   +--9 (parallel)<br />


--------------------------------------------------------------------------------<br />
------------------------------ After Optimisation ------------------------------<br />
Parallel region 0:<br />
+--10 (parallel)<br />
   +--9 (serial)<br />


 
Parallel region 0 (loop #10) had 0 loop(s) fused and 1 loop(s) serialized as <br />
part of the larger parallel loop (#10).<br />
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
 
---------------------------Loop invariant code motion---------------------------<br />
Allocation hoisting:<br />
The memory allocation derived from the instruction at <br />
/Users/nathaniel/Desktop/MLE/workspace/mle-<br />
module-3-Nathaniel-Nirmal/minitorch/fast_ops.py (291) is hoisted out of the parallel <br />
loop labelled #10 (it will be performed before the loop is executed and reused <br />
inside the loop):<br />
   Allocation:: out_index: Index = np.zeros(MAX_DIMS, np.int32)<br />
    - numpy.empty() is used for the allocation.
None<br />
MATRIX MULTIPLY<br />
 
================================================================================
 Parallel Accelerator Optimizing:  Function _tensor_matrix_multiply, <br />
/Users/nathaniel/Desktop/MLE/workspace/mle-<br />
module-3-Nathaniel-Nirmal/minitorch/fast_ops.py (303)  <br />
================================================================================


Parallel loop listing for  Function _tensor_matrix_multiply, /Users/nathaniel/Desktop/MLE/workspace/mle-module-3-Nathaniel-Nirmal/minitorch/fast_ops.py (303) 
-----------------------------------------------------------------------------------------|loop #ID
def _tensor_matrix_multiply(                                                             | <br />
    out: Storage,                                                                        | <br />
    out_shape: Shape,                                                                    | <br />
    out_strides: Strides,                                                                | <br />
    a_storage: Storage,                                                                  | <br />
    a_shape: Shape,                                                                      | <br />
    a_strides: Strides,                                                                  | <br />
    b_storage: Storage,                                                                  | <br />
    b_shape: Shape,                                                                      | <br />
    b_strides: Strides,                                                                  | <br />
) -> None:                                                                               | <br />
    """                                                                                  | <br />
    NUMBA tensor matrix multiply function.                                               | <br />
                                                                                         | <br />
    Should work for any tensor shapes that broadcast as long as                          | <br />
                                                                                         | <br />
    ```                                                                                  | <br />
    assert a_shape[-1] == b_shape[-2]                                                    | <br />
    ```                                                                                  | <br />
                                                                                         | <br />
    Optimizations:                                                                       | <br />
                                                                                         | <br />
    * No index buffers or function calls                                                 | <br />
    * Inner loop should have no global writes, 1 multiply.                               | <br />
                                                                                         | <br />
                                                                                         | <br />
    Args:                                                                                | <br />
        out (Storage): storage for `out` tensor                                          | <br />
        out_shape (Shape): shape for `out` tensor                                        | <br />
        out_strides (Strides): strides for `out` tensor                                  | <br />
        a_storage (Storage): storage for `a` tensor                                      | <br />
        a_shape (Shape): shape for `a` tensor                                            | <br />
        a_strides (Strides): strides for `a` tensor                                      | <br />
        b_storage (Storage): storage for `b` tensor                                      | <br />
        b_shape (Shape): shape for `b` tensor                                            | <br />
        b_strides (Strides): strides for `b` tensor                                      | <br />
                                                                                         | <br />
    Returns:                                                                             | <br />
        None : Fills in `out`                                                            | <br />
    """                                                                                  | <br />
    # TODO: Implement for Task 3.2.                                                      | <br />
    a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0                               | <br />
    b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0                               | <br />
                                                                                         | <br />
    # TODO: Implement for Task 3.2.                                                      | <br />
    # Looping through each of the output indices                                         | <br />
    for i in prange(out_shape[0]):-------------------------------------------------------| #13<br />
        for j in prange(out_shape[1]):---------------------------------------------------| #12<br />
            for k in prange(out_shape[2]):-----------------------------------------------| #11<br />
                val = 0.0                                                                | <br />
                p_a = i * a_batch_stride + j * a_strides[1]                              | <br />
                p_b = i * b_batch_stride + k * b_strides[2]                              | <br />
                for _ in range(a_shape[2]):                                              | <br />
                    val += a_storage[p_a] * b_storage[p_b]                               | <br />
                    p_a += a_strides[2]                                                  | <br />
                    p_b += b_strides[1]                                                  | <br />
                outPos = i * out_strides[0] + j * out_strides[1] + k * out_strides[2]    | <br />
                out[outPos] = val                                                        | <br />
--------------------------------- Fusing loops ---------------------------------<br />
Attempting fusion of parallel loops (combines loops with similar properties)...<br />
Following the attempted fusion of parallel for-loops there are 2 parallel for-<br />
loop(s) (originating from loops labelled: #13, #12).<br />
--------------------------------------------------------------------------------
---------------------------- Optimising loop nests -----------------------------<br />
Attempting loop nest rewrites (optimising for the largest parallel loops)...<br />
 
+--13 is a parallel loop<br />
   +--12 --> rewritten as a serial loop<br />
      +--11 --> rewritten as a serial loop<br />
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------<br />
Parallel region 0:<br />
+--13 (parallel)<br />
   +--12 (parallel)<br />
      +--11 (parallel)<br />


--------------------------------------------------------------------------------<br />
------------------------------ After Optimisation ------------------------------<br />
Parallel region 0:<br />
+--13 (parallel)<br />
   +--12 (serial)<br />
      +--11 (serial)<br />


 
Parallel region 0 (loop #13) had 0 loop(s) fused and 2 loop(s) serialized as <br />
part of the larger parallel loop (#13).<br />
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
 
---------------------------Loop invariant code motion---------------------------<br />
Allocation hoisting:<br />
No allocation hoisting found<br />
None<br />



### run_fast_tensor GPU Hidden=100 Dataset=split
!cd $DIR; PYTHONPATH=/content/$DIR python3.8 project/run_fast_tensor.py --BACKEND gpu --HIDDEN 100 --DATASET split --RATE 0.05<br />
Epoch  0  loss  5.726708154806323 correct 37 time  0.7933566570281982<br />
Epoch  10  loss  4.380296642174042 correct 43 time  2.898868441581726<br />
Epoch  20  loss  3.609300068449616 correct 44 time  2.3746673107147216<br />
Epoch  30  loss  4.409434020278958 correct 48 time  2.393710803985596<br />
Epoch  40  loss  2.348035174581344 correct 48 time  2.3780134439468386<br />
Epoch  50  loss  2.645560793506277 correct 44 time  2.382448601722717<br />
Epoch  60  loss  1.3970239729743137 correct 47 time  2.385243272781372<br />
Epoch  70  loss  1.729701943119497 correct 49 time  2.5842482328414915<br />
Epoch  80  loss  1.2305072369147412 correct 49 time  2.627967619895935<br />
Epoch  90  loss  0.7340348885263437 correct 49 time  2.379882574081421<br />
Epoch  100  loss  1.1182476300573407 correct 49 time  2.387559151649475<br />
Epoch  110  loss  0.942253010104005 correct 49 time  2.3870112180709837<br />
Epoch  120  loss  1.1398925973142746 correct 49 time  2.3914215087890627<br />
Epoch  130  loss  0.32920792924479597 correct 49 time  2.4066972494125367<br />
Epoch  140  loss  0.576432853863463 correct 49 time  2.75266489982605<br />
Epoch  150  loss  0.6696496962719144 correct 50 time  2.490497589111328<br />
Epoch  160  loss  0.7374211089721112 correct 48 time  2.3867936611175535<br />
Epoch  170  loss  1.3976982566447593 correct 48 time  2.380331349372864<br />
Epoch  180  loss  0.6312755768662945 correct 48 time  2.387139821052551<br />
Epoch  190  loss  0.706281130761407 correct 48 time  2.3959224939346315<br />
Epoch  200  loss  1.4817999912855095 correct 50 time  2.3732172727584837<br />
Epoch  210  loss  1.3354451703465555 correct 50 time  2.5667962551116945<br />
Epoch  220  loss  0.1637685122695534 correct 49 time  2.3865289449691773<br />
Epoch  230  loss  0.4912180010348279 correct 48 time  2.3925663232803345<br />
Epoch  240  loss  1.1661813486092307 correct 50 time  2.366002082824707<br />
Epoch  250  loss  0.2976131904386084 correct 49 time  2.412809228897095<br />
Epoch  260  loss  0.26840803864635115 correct 49 time  2.386847639083862<br />
Epoch  270  loss  0.8532418295482805 correct 49 time  2.3811151504516603<br />
Epoch  280  loss  0.38445748252607087 correct 49 time  2.836422157287598<br />
Epoch  290  loss  0.11177732071778589 correct 49 time  2.4359109163284303<br />
Epoch  300  loss  2.6455409505222645 correct 47 time  2.389882135391235<br />
Epoch  310  loss  1.1893662106646699 correct 49 time  2.3581416845321654<br />
Epoch  320  loss  0.712856947504082 correct 49 time  2.405737829208374<br />
Epoch  330  loss  0.4683957166945902 correct 46 time  2.3695279359817505<br />
Epoch  340  loss  0.2527447850683378 correct 49 time  2.896191048622131<br />
Epoch  350  loss  0.7595892904684916 correct 49 time  2.3720913410186766<br />
Epoch  360  loss  0.9441415003121056 correct 50 time  2.3667574644088747<br />
Epoch  370  loss  1.3921706665170197 correct 50 time  2.379753756523132<br />
Epoch  380  loss  1.1172836960568264 correct 49 time  2.395392918586731<br />
Epoch  390  loss  0.7046780220940144 correct 50 time  2.3981858015060427<br />
Epoch  400  loss  0.6538499279905563 correct 50 time  2.354160451889038<br />
Epoch  410  loss  0.15809345606273822 correct 50 time  2.3606741428375244<br />
Epoch  420  loss  0.7163933932299265 correct 50 time  2.643211078643799<br />
Epoch  430  loss  0.09216634714687053 correct 50 time  2.3934902429580687<br />
Epoch  440  loss  0.4418224014408654 correct 49 time  2.366760277748108<br />
Epoch  450  loss  0.36930979987608875 correct 49 time  2.3638375282287596<br />
Epoch  460  loss  2.328848993325287 correct 48 time  2.366055488586426<br />
Epoch  470  loss  0.2875127472091348 correct 50 time  2.3553855657577514<br />
Epoch  480  loss  0.20751174297188388 correct 48 time  3.0435333013534547<br />
Epoch  490  loss  1.015132009472366 correct 50 time  2.3727704763412474<br />


### run_fast_tensor GPU Hidden=100 Dataset=simple<br />
!cd $DIR; PYTHONPATH=/content/$DIR python3.8 project/run_fast_tensor.py --BACKEND gpu --HIDDEN 100 --DATASET simple --RATE 0.05<br />
Epoch  0  loss  4.628345668788533 correct 38 time  0.786777663230896<br />
Epoch  10  loss  3.4560930200589137 correct 46 time  2.3444525718688967<br />
Epoch  20  loss  1.6271680392467982 correct 48 time  3.1506524085998535<br />
Epoch  30  loss  2.209267113215622 correct 50 time  2.3566426753997805<br />
Epoch  40  loss  1.3800075490785264 correct 48 time  2.35113844871521<br />
Epoch  50  loss  0.4925438352053896 correct 49 time  2.373103070259094<br />
Epoch  60  loss  1.2310837811531952 correct 49 time  2.3423258304595946<br />
Epoch  70  loss  0.7116183049405407 correct 50 time  2.3674262046813963<br />
Epoch  80  loss  0.3244943928745453 correct 49 time  2.34915828704834<br />
Epoch  90  loss  1.3620330077076597 correct 46 time  2.3511711597442626<br />
Epoch  100  loss  0.7593938259627585 correct 50 time  2.345990753173828<br />
Epoch  110  loss  1.4795255478180735 correct 49 time  2.8889660596847535<br />
Epoch  120  loss  0.4660727542827543 correct 50 time  2.3835933208465576<br />
Epoch  130  loss  0.4184724567436816 correct 49 time  2.3887112617492674<br />
Epoch  140  loss  0.6587585104509114 correct 49 time  2.385755515098572<br />
Epoch  150  loss  1.2021869826156895 correct 50 time  2.4734914779663084<br />
Epoch  160  loss  0.5797765755039257 correct 49 time  2.76632182598114<br />
Epoch  170  loss  0.9544049437065903 correct 50 time  2.360326886177063<br />
Epoch  180  loss  1.1445087886929315 correct 50 time  2.3615612983703613<br />
Epoch  190  loss  0.23904930653223058 correct 49 time  2.359925413131714<br />
Epoch  200  loss  0.623348306762042 correct 50 time  2.3690621614456178<br />
Epoch  210  loss  0.43322123640565335 correct 50 time  2.3635371208190916<br />
Epoch  220  loss  0.584788996634706 correct 50 time  2.3731263637542725<br />
Epoch  230  loss  0.26946837905986654 correct 50 time  2.362049698829651<br />
Epoch  240  loss  0.3116273558629678 correct 50 time  2.3478962182998657<br />
Epoch  250  loss  1.5825751434181694 correct 48 time  2.5511038303375244<br />
Epoch  260  loss  0.2096210767712041 correct 50 time  2.3600850820541384<br />
Epoch  270  loss  0.12353309819408653 correct 50 time  2.384626317024231<br />
Epoch  280  loss  0.999795330443415 correct 50 time  2.3581949949264525<br />
Epoch  290  loss  0.6785004233762926 correct 50 time  2.575086760520935<br />
Epoch  300  loss  0.8511963311279808 correct 49 time  2.641939973831177<br />
Epoch  310  loss  0.09826832061430318 correct 50 time  2.3686970233917237<br />
Epoch  320  loss  0.0006959368998094348 correct 50 time  2.4390244483947754<br />
Epoch  330  loss  0.7041468724751416 correct 49 time  2.389383625984192<br />
Epoch  340  loss  0.11990284261216508 correct 50 time  2.396605038642883<br />
Epoch  350  loss  0.20938089971773244 correct 50 time  2.364039397239685<br />
Epoch  360  loss  0.4082665161210739 correct 50 time  2.3620493173599244<br />
Epoch  370  loss  0.2947185850996204 correct 50 time  2.3595596075057985<br />
Epoch  380  loss  0.009909838035355728 correct 50 time  2.362058186531067<br />
Epoch  390  loss  0.14045461198172116 correct 50 time  2.872450566291809<br />
Epoch  400  loss  0.16837392842450893 correct 50 time  2.330647921562195<br />
Epoch  410  loss  0.24105766385381902 correct 50 time  2.3907458066940306<br />
Epoch  420  loss  0.42810262942062705 correct 49 time  2.3702556848526<br />
Epoch  430  loss  0.034736495312726386 correct 49 time  2.9456573724746704<br />
Epoch  440  loss  0.014096159462298388 correct 50 time  2.3647982120513915<br />
Epoch  450  loss  0.10191580575012464 correct 49 time  2.3678321838378906<br />
Epoch  460  loss  0.0008204085534406931 correct 49 time  2.3390871047973634<br />
Epoch  470  loss  0.0020896695961135836 correct 50 time  2.3485584259033203<br />
Epoch  480  loss  0.6995874233974125 correct 50 time  2.401357126235962<br />
Epoch  490  loss  0.20272709099095784 correct 50 time  2.3692042350769045<br />



### run_fast_tensor GPU Hidden=100 Dataset=diag<br />
!cd $DIR; PYTHONPATH=/content/$DIR python3.8 project/run_fast_tensor.py --BACKEND gpu --HIDDEN 100 --DATASET diag --RATE 0.05<br />
Epoch  0  loss  4.909212464692095 correct 41 time  0.3926310777664185<br />
Epoch  10  loss  1.6385467471850996 correct 47 time  2.369743013381958<br />
Epoch  20  loss  0.6598509750420949 correct 50 time  2.3991901159286497<br />
Epoch  30  loss  0.7498399738344826 correct 50 time  2.3440657377243044<br />
Epoch  40  loss  0.6543736714367865 correct 50 time  2.3683050632476808<br />
Epoch  50  loss  0.3577176382847036 correct 50 time  2.3598959922790526<br />
Epoch  60  loss  0.37094188376216397 correct 50 time  2.372000288963318<br />
Epoch  70  loss  0.19443344099787244 correct 50 time  2.3738095998764037<br />
Epoch  80  loss  0.44985595334237793 correct 50 time  2.76826388835907<br />
Epoch  90  loss  0.7134921331709623 correct 50 time  2.4534961462020872<br />
Epoch  100  loss  0.11171504488457587 correct 50 time  2.3960694313049316<br />
Epoch  110  loss  0.21980391724376985 correct 50 time  2.392460584640503<br />
Epoch  120  loss  0.3311695192467129 correct 50 time  2.57126145362854<br />
Epoch  130  loss  0.502387750611146 correct 50 time  2.367454195022583<br />
Epoch  140  loss  0.36303566114032754 correct 50 time  2.365046811103821<br />
Epoch  150  loss  0.504661581313938 correct 50 time  2.367484450340271<br />
Epoch  160  loss  0.3216366730494607 correct 50 time  2.3746177673339846<br />
Epoch  170  loss  0.025389722250497182 correct 50 time  2.404364824295044<br />
Epoch  180  loss  0.009178954203856574 correct 50 time  2.4276166915893556<br />
Epoch  190  loss  0.15038043517577793 correct 50 time  2.3937432527542115<br />
Epoch  200  loss  0.36630773963847674 correct 50 time  2.382674312591553<br />
Epoch  210  loss  0.17217302983196336 correct 50 time  2.3895465612411497<br />
Epoch  220  loss  0.23519547871363838 correct 50 time  2.397544527053833<br />
Epoch  230  loss  0.0852709526944436 correct 50 time  2.6188247203826904<br />
Epoch  240  loss  0.20912834409913122 correct 50 time  2.3653831481933594<br />
Epoch  250  loss  0.2970370884192952 correct 50 time  2.4054778099060057<br />
Epoch  260  loss  0.0029540991069540523 correct 50 time  2.585608148574829<br />
Epoch  270  loss  0.0028865061107412205 correct 50 time  2.3666165113449096<br />
Epoch  280  loss  0.21418913573103152 correct 50 time  2.3669606924057005<br />
Epoch  290  loss  0.20296792282228532 correct 50 time  2.358390212059021<br />
Epoch  300  loss  0.012326885371340014 correct 50 time  2.360761857032776<br />
Epoch  310  loss  0.13268088201262007 correct 50 time  2.3692918539047243<br />
Epoch  320  loss  0.2048416443710311 correct 50 time  2.3994604110717774<br />
Epoch  330  loss  0.04210458430218873 correct 50 time  2.3576626062393187<br />
Epoch  340  loss  0.04433452816366119 correct 50 time  2.360717606544495<br />
Epoch  350  loss  0.16859213664311856 correct 50 time  2.3553906202316286<br />
Epoch  360  loss  0.036612410245223714 correct 50 time  2.369006848335266<br />
Epoch  370  loss  0.1810088908703126 correct 50 time  2.8214903354644774<br />
Epoch  380  loss  0.12431740406796037 correct 50 time  2.3500909566879273<br />
Epoch  390  loss  0.013814786755817833 correct 50 time  2.352052593231201<br />
Epoch  400  loss  0.0007454754817393796 correct 50 time  3.026097846031189<br />
Epoch  410  loss  0.22295996974287952 correct 50 time  2.3493371963500977<br />
Epoch  420  loss  0.19921502913659841 correct 50 time  2.354534959793091<br />
Epoch  430  loss  0.008139964963051208 correct 50 time  2.3244752645492555<br />
Epoch  440  loss  0.21956560779784348 correct 50 time  2.3398024797439576<br />
Epoch  450  loss  0.132269123646013 correct 50 time  2.344083476066589<br />
Epoch  460  loss  0.13466644324521426 correct 50 time  2.3643606662750245<br />
Epoch  470  loss  0.05424826751818724 correct 50 time  2.3649816036224367<br />
Epoch  480  loss  0.060306601155991284 correct 50 time  2.340773677825928<br />
Epoch  490  loss  0.09402958934067664 correct 50 time  2.370497488975525<br />



### run_fast_tensor GPU Hidden=100 Dataset=xor<br />
!cd $DIR; PYTHONPATH=/content/$DIR python3.8 project/run_fast_tensor.py --BACKEND gpu --HIDDEN 100 --DATASET xor --RATE 0.05<br />
Epoch  0  loss  7.4728676442851105 correct 34 time  0.4371016502380371<br />
Epoch  10  loss  5.788671815757721 correct 44 time  2.367793560028076<br />
Epoch  20  loss  3.0855620207380396 correct 46 time  2.358235263824463<br />
Epoch  30  loss  2.4943659815672907 correct 47 time  2.349768567085266<br />
Epoch  40  loss  2.3920906613649224 correct 46 time  2.3673400402069094<br />
Epoch  50  loss  4.0041487020390285 correct 46 time  2.3503172397613525<br />
Epoch  60  loss  1.0212419986742312 correct 47 time  2.3643386125564576<br />
Epoch  70  loss  2.0118394287322823 correct 46 time  2.391431140899658<br />
Epoch  80  loss  2.5042263496489565 correct 48 time  2.3588892459869384<br />
Epoch  90  loss  3.0886700538030745 correct 46 time  2.355507564544678<br />
Epoch  100  loss  1.2967148916763218 correct 46 time  2.369547128677368<br />
Epoch  110  loss  1.0409547160854695 correct 48 time  2.922721290588379<br />
Epoch  120  loss  1.5628561192915063 correct 48 time  2.357061171531677<br />
Epoch  130  loss  1.6478821499454093 correct 47 time  2.7751298189163207<br />
Epoch  140  loss  1.3365286748905933 correct 48 time  2.3572976350784303<br />
Epoch  150  loss  0.627391337026526 correct 48 time  2.373391556739807<br />
Epoch  160  loss  2.252295023638983 correct 48 time  2.3545324087142943<br />
Epoch  170  loss  1.9764156931944648 correct 49 time  2.3869351625442503<br />
Epoch  180  loss  1.5896703909076701 correct 49 time  2.3317782878875732<br />
Epoch  190  loss  1.8156680089673998 correct 48 time  2.334757089614868<br />
Epoch  200  loss  1.6686905999509865 correct 49 time  2.3593642950057983<br />
Epoch  210  loss  1.4599674557626243 correct 48 time  2.3567100524902345<br />
Epoch  220  loss  2.0496573486869845 correct 49 time  2.3627736568450928<br />
Epoch  230  loss  1.495971037693597 correct 49 time  2.3748227834701536<br />
Epoch  240  loss  0.3901723879450936 correct 49 time  2.3419420957565307<br />
Epoch  250  loss  1.2393472122365867 correct 49 time  2.3446494340896606<br />
Epoch  260  loss  0.43932196652453975 correct 49 time  2.8597918272018434<br />
Epoch  270  loss  0.42623733539093367 correct 47 time  2.3730397701263426<br />
Epoch  280  loss  1.752169807592652 correct 50 time  2.6840341329574584<br />
Epoch  290  loss  0.6460727162052117 correct 48 time  2.344062161445618<br />
Epoch  300  loss  1.247266516725839 correct 48 time  2.40957658290863<br />
Epoch  310  loss  0.6453075550806681 correct 47 time  2.345094633102417<br />
Epoch  320  loss  1.6608768478190399 correct 49 time  2.349836254119873<br />
Epoch  330  loss  1.6201886905587495 correct 48 time  2.331174874305725<br />
Epoch  340  loss  1.1149707590663533 correct 50 time  2.362939977645874<br />
Epoch  350  loss  0.36748070420666085 correct 50 time  2.3487688064575196<br />
Epoch  360  loss  0.5243944242664438 correct 49 time  2.3539325475692747<br />
Epoch  370  loss  0.944603126183351 correct 50 time  2.345224666595459<br />
Epoch  380  loss  1.9179527264572003 correct 49 time  2.356510782241821<br />
Epoch  390  loss  0.4681671434614694 correct 49 time  2.3430718660354612<br />
Epoch  400  loss  1.7286362843002376 correct 49 time  2.6397414207458496<br />
Epoch  410  loss  0.9462065475578 correct 48 time  2.36535758972168<br />
Epoch  420  loss  1.0180091835302156 correct 49 time  2.647882866859436<br />
Epoch  430  loss  1.6126505680804237 correct 50 time  2.3284967660903932<br />
Epoch  440  loss  0.045603276278697175 correct 49 time  2.350307822227478<br />
Epoch  450  loss  0.14346921176306435 correct 49 time  2.3478535652160644<br />
Epoch  460  loss  0.2793342642523006 correct 49 time  2.325368618965149<br />
Epoch  470  loss  1.4554992180552953 correct 48 time  2.3844754695892334<br />
Epoch  480  loss  0.392222150720549 correct 48 time  2.3572187662124633<br />
Epoch  490  loss  1.1165849045986964 correct 48 time  2.3651877880096435<br />



### run_fast_tensor CPU Hidden=100 Dataset=split<br />
!cd $DIR; PYTHONPATH=/content/$DIR python3.8 project/run_fast_tensor.py --BACKEND cpu --HIDDEN 100 --DATASET split --RATE 0.05<br />
Epoch  0  loss  7.289760050266265 correct 31 time  2.9496093511581423<br />
Epoch  10  loss  6.065942881185627 correct 43 time  0.10593204498291016<br />
Epoch  20  loss  5.840326915014369 correct 42 time  0.10358116626739503<br />
Epoch  30  loss  4.367456881023578 correct 46 time  0.1043656587600708<br />
Epoch  40  loss  3.6078403601866778 correct 47 time  0.10269386768341064<br />
Epoch  50  loss  2.025190340825894 correct 46 time  0.21303038597106932<br />
Epoch  60  loss  2.468451423152021 correct 45 time  0.20842332839965821<br />
Epoch  70  loss  1.1143396766351095 correct 42 time  0.23698346614837645<br />
Epoch  80  loss  1.8799765855769064 correct 44 time  0.24729819297790528<br />
Epoch  90  loss  2.148341359434963 correct 44 time  0.2505024909973145<br />
Epoch  100  loss  3.0633005286561015 correct 49 time  0.23050415515899658<br />
Epoch  110  loss  2.8347404840233987 correct 47 time  0.10218415260314942<br />
Epoch  120  loss  2.4096398830987513 correct 49 time  0.10139930248260498<br />
Epoch  130  loss  1.5119567431557674 correct 48 time  0.1060915470123291<br />
Epoch  140  loss  1.7389949661339028 correct 49 time  0.10438108444213867<br />
Epoch  150  loss  0.2918076806670313 correct 47 time  0.10193657875061035<br />
Epoch  160  loss  1.0073300752570078 correct 49 time  0.10243964195251465<br />
Epoch  170  loss  1.3616189304870334 correct 50 time  0.1026010274887085<br />
Epoch  180  loss  2.3624175909611207 correct 49 time  0.10209245681762695<br />
Epoch  190  loss  1.1381920527846883 correct 47 time  0.10135951042175292<br />
Epoch  200  loss  2.592714170725853 correct 48 time  0.11415021419525147<br />
Epoch  210  loss  1.901908936829921 correct 46 time  0.21635050773620607<br />
Epoch  220  loss  2.246409173517006 correct 50 time  0.20323398113250732<br />
Epoch  230  loss  1.5645959167828445 correct 46 time  0.25175430774688723<br />
Epoch  240  loss  2.5368119590276335 correct 47 time  0.2542442798614502<br />
Epoch  250  loss  0.25813447499809783 correct 50 time  0.2524466276168823<br />
Epoch  260  loss  2.0156247838134203 correct 50 time  0.21342220306396484<br />
Epoch  270  loss  0.344252181641017 correct 46 time  0.1341519832611084<br />
Epoch  280  loss  5.388476085724657 correct 44 time  0.10578994750976563<br />
Epoch  290  loss  0.7886403779425221 correct 49 time  0.10309135913848877<br />
Epoch  300  loss  1.5646461765992496 correct 46 time  0.10314111709594727<br />
Epoch  310  loss  0.8189982258462739 correct 50 time  0.10278043746948243<br />
Epoch  320  loss  0.3665337431494248 correct 47 time  0.10480248928070068<br />
Epoch  330  loss  1.758406009711332 correct 45 time  0.10167005062103271<br />
Epoch  340  loss  2.8889841189463072 correct 47 time  0.10589802265167236<br />
Epoch  350  loss  1.258369314735394 correct 47 time  0.10416781902313232<br />
Epoch  360  loss  0.33594431419555426 correct 50 time  0.20227177143096925<br />
Epoch  370  loss  0.4192025475669574 correct 48 time  0.21351752281188965<br />
Epoch  380  loss  2.1933725599049105 correct 47 time  0.2203078031539917<br />
Epoch  390  loss  1.0369614505040226 correct 47 time  0.2369532823562622<br />
Epoch  400  loss  1.8775433050669583 correct 48 time  0.24436824321746825<br />
Epoch  410  loss  1.5659126334481273 correct 49 time  0.24799776077270508<br />
Epoch  420  loss  1.3926939399025893 correct 50 time  0.12417173385620117<br />
Epoch  430  loss  1.180383934632444 correct 49 time  0.10363879203796386<br />
Epoch  440  loss  0.4145007046358194 correct 50 time  0.10270817279815674<br />
Epoch  450  loss  1.6963721719253302 correct 50 time  0.10539970397949219<br />
Epoch  460  loss  0.12444927605797301 correct 50 time  0.10617337226867676<br />
Epoch  470  loss  1.6925152309140037 correct 50 time  0.10441460609436035<br />
Epoch  480  loss  0.9760403796915235 correct 50 time  0.10270626544952392<br />
Epoch  490  loss  0.524777463613874 correct 49 time  0.10237736701965332<br />



### run_fast_tensor CPU Hidden=100 Dataset=simple<br />
!cd $DIR; PYTHONPATH=/content/$DIR python3.8 project/run_fast_tensor.py --BACKEND cpu --HIDDEN 100 --DATASET simple --RATE 0.05<br />
Epoch  0  loss  4.984244774295874 correct 45 time  3.1012641191482544<br />
Epoch  10  loss  1.5258745916087166 correct 49 time  0.21019153594970702<br />
Epoch  20  loss  1.5995908049311645 correct 47 time  0.2492516040802002<br />
Epoch  30  loss  0.36523234306753466 correct 49 time  0.2412137508392334<br />
Epoch  40  loss  0.24650908523844264 correct 48 time  0.25468766689300537<br />
Epoch  50  loss  0.34769209015478686 correct 50 time  0.22268764972686766<br />
Epoch  60  loss  0.20230031976529222 correct 50 time  0.10161776542663574<br />
Epoch  70  loss  1.2526202299071456 correct 50 time  0.10263597965240479<br />
Epoch  80  loss  0.7694006150853059 correct 50 time  0.10182573795318603<br />
Epoch  90  loss  0.9823990661807619 correct 50 time  0.10173702239990234<br />
Epoch  100  loss  0.842287939477482 correct 50 time  0.10303893089294433<br />
Epoch  110  loss  0.43360491651529076 correct 50 time  0.10249242782592774<br />
Epoch  120  loss  0.014148864726584465 correct 50 time  0.10630834102630615<br />
Epoch  130  loss  0.5264128938781658 correct 50 time  0.10537829399108886<br />
Epoch  140  loss  0.4011973729632591 correct 50 time  0.10225908756256104<br />
Epoch  150  loss  0.3479389315678171 correct 50 time  0.13973646163940429<br />
Epoch  160  loss  0.633879432821367 correct 50 time  0.21447618007659913<br />
Epoch  170  loss  0.7131158203674666 correct 50 time  0.22550780773162843<br />
Epoch  180  loss  0.0361311746136837 correct 50 time  0.2383345127105713<br />
Epoch  190  loss  0.14334202952370437 correct 50 time  0.24373631477355956<br />
Epoch  200  loss  0.09274924382955337 correct 50 time  0.24792020320892333<br />
Epoch  210  loss  0.029295033916571296 correct 50 time  0.19766101837158204<br />
Epoch  220  loss  0.038569295766965835 correct 50 time  0.10451593399047851<br />
Epoch  230  loss  0.17041464968831013 correct 50 time  0.1008638858795166<br />
Epoch  240  loss  0.36040274106099596 correct 50 time  0.10037832260131836<br />
Epoch  250  loss  0.11404495034140717 correct 50 time  0.10109152793884277<br />
Epoch  260  loss  0.08898016678556935 correct 50 time  0.10194792747497558<br />
Epoch  270  loss  0.3582767092553263 correct 50 time  0.10103793144226074<br />
Epoch  280  loss  0.028175049324217424 correct 50 time  0.10109853744506836<br />
Epoch  290  loss  0.026644387951707905 correct 50 time  0.10111150741577149<br />
Epoch  300  loss  0.11169124803778842 correct 50 time  0.10152115821838378<br />
Epoch  310  loss  0.29935887092405633 correct 50 time  0.16828668117523193<br />
Epoch  320  loss  0.44613050990026304 correct 50 time  0.2120438814163208<br />
Epoch  330  loss  0.15435758755603265 correct 50 time  0.22803144454956054<br />
Epoch  340  loss  0.26431784516236506 correct 50 time  0.25256597995758057<br />
Epoch  350  loss  0.0928896753220904 correct 50 time  0.2498568058013916<br />
Epoch  360  loss  0.07409903582979632 correct 50 time  0.25638115406036377<br />
Epoch  370  loss  0.023975479369199477 correct 50 time  0.1378178358078003<br />
Epoch  380  loss  0.037616969534262724 correct 50 time  0.10194265842437744<br />
Epoch  390  loss  0.30708733657182413 correct 50 time  0.10195002555847169<br />
Epoch  400  loss  0.0924480131026035 correct 50 time  0.10189275741577149<br />
Epoch  410  loss  0.030407509505197973 correct 50 time  0.10185790061950684<br />
Epoch  420  loss  0.00577203626611756 correct 50 time  0.10083811283111573<br />
Epoch  430  loss  0.30097320441516573 correct 50 time  0.1033095359802246<br />
Epoch  440  loss  0.03126102677014727 correct 50 time  0.1036895751953125<br />
Epoch  450  loss  0.01768058199351859 correct 50 time  0.10259168148040772<br />
Epoch  460  loss  0.1050641148534921 correct 50 time  0.10231785774230957<br />
Epoch  470  loss  0.0008198323708754833 correct 50 time  0.1981750249862671<br />
Epoch  480  loss  0.02128001121730009 correct 50 time  0.20055346488952636<br />
Epoch  490  loss  0.2490204301060333 correct 50 time  0.24960644245147706<br />



### run_fast_tensor CPU Hidden=100 Dataset=diag<br />
!cd $DIR; PYTHONPATH=/content/$DIR python3.8 project/run_fast_tensor.py --BACKEND cpu --HIDDEN 100 --DATASET diag --RATE 0.05<br />
Epoch  0  loss  3.052622831798379 correct 43 time  2.9436318397521974<br />
Epoch  10  loss  2.571145041447419 correct 49 time  0.10166890621185302<br />
Epoch  20  loss  0.938898231438188 correct 49 time  0.1015702486038208<br />
Epoch  30  loss  0.13075491870082212 correct 50 time  0.10239219665527344<br />
Epoch  40  loss  1.1304086486064715 correct 50 time  0.11502654552459717<br />
Epoch  50  loss  1.185468064553183 correct 50 time  0.21135416030883789<br />
Epoch  60  loss  0.32336781212690796 correct 50 time  0.22362051010131836<br />
Epoch  70  loss  1.022083289760516 correct 50 time  0.22328658103942872<br />
Epoch  80  loss  0.5413904187023353 correct 50 time  0.23838555812835693<br />
Epoch  90  loss  0.10385118035955093 correct 50 time  0.23339438438415527<br />
Epoch  100  loss  0.19625192796838975 correct 50 time  0.23453261852264404<br />
Epoch  110  loss  0.018093358228996066 correct 50 time  0.10130131244659424<br />
Epoch  120  loss  0.16268435371771672 correct 50 time  0.10160288810729981<br />
Epoch  130  loss  0.40862754964557146 correct 50 time  0.11020188331604004<br />
Epoch  140  loss  0.7298564990306801 correct 50 time  0.11001300811767578<br />
Epoch  150  loss  0.38994494755480386 correct 50 time  0.10623817443847657<br />
Epoch  160  loss  0.2951634140462562 correct 50 time  0.10303561687469483<br />
Epoch  170  loss  0.005399480696693092 correct 50 time  0.10484368801116943<br />
Epoch  180  loss  0.05105122355159346 correct 50 time  0.10679728984832763<br />
Epoch  190  loss  0.48394214078472 correct 50 time  0.1026501178741455<br />
Epoch  200  loss  0.006043571345592764 correct 50 time  0.1498739242553711<br />
Epoch  210  loss  0.025127917988205725 correct 50 time  0.2289274215698242<br />
Epoch  220  loss  0.3263641809178931 correct 50 time  0.225252103805542<br />
Epoch  230  loss  0.2999451022611734 correct 50 time  0.23494598865509034<br />
Epoch  240  loss  0.11938423566924493 correct 50 time  0.2485027313232422<br />
Epoch  250  loss  0.05218210010502074 correct 50 time  0.2235950708389282<br />
Epoch  260  loss  0.06754547185684937 correct 50 time  0.20506150722503663<br />
Epoch  270  loss  0.4958273565792867 correct 50 time  0.10995721817016602<br />
Epoch  280  loss  0.015531516807082996 correct 50 time  0.11072473526000977<br />
Epoch  290  loss  0.05410045486761421 correct 50 time  0.10803892612457275<br />
Epoch  300  loss  0.0016070884434588996 correct 50 time  0.10727560520172119<br />
Epoch  310  loss  0.018436790291058304 correct 50 time  0.1047583818435669<br />
Epoch  320  loss  0.05876487573259296 correct 50 time  0.10366206169128418<br />
Epoch  330  loss  0.269762063718975 correct 50 time  0.10638434886932373<br />
Epoch  340  loss  0.48217533626033976 correct 50 time  0.1027489423751831<br />
Epoch  350  loss  0.03152551526584689 correct 50 time  0.1112062931060791<br />
Epoch  360  loss  0.059847272010608826 correct 50 time  0.21746344566345216<br />
Epoch  370  loss  0.02514231661409725 correct 50 time  0.21821627616882325<br />
Epoch  380  loss  0.25689368048235345 correct 50 time  0.2393026351928711<br />
Epoch  390  loss  0.009286854027459253 correct 50 time  0.2456160306930542<br />
Epoch  400  loss  0.001676339542447712 correct 50 time  0.2221090316772461<br />
Epoch  410  loss  0.0016159863473526952 correct 50 time  0.23980176448822021<br />
Epoch  420  loss  0.05605467034786564 correct 50 time  0.11747510433197021<br />
Epoch  430  loss  0.27523392539599034 correct 50 time  0.10658786296844483<br />
Epoch  440  loss  0.03403945209981528 correct 50 time  0.1069526195526123<br />
Epoch  450  loss  0.029672915071696195 correct 50 time  0.1105112075805664<br />
Epoch  460  loss  0.25093355531345746 correct 50 time  0.10827672481536865<br />
Epoch  470  loss  0.23314218845073673 correct 50 time  0.1076587438583374<br />
Epoch  480  loss  0.4200410017437295 correct 50 time  0.10692067146301269<br />
Epoch  490  loss  0.0052700427854177456 correct 50 time  0.1080888032913208<br />



### run_fast_tensor CPU Hidden=100 Dataset=xor<br />
!cd $DIR; PYTHONPATH=/content/$DIR python3.8 project/run_fast_tensor.py --BACKEND cpu --HIDDEN 100 --DATASET xor --RATE 0.05<br />
Epoch  0  loss  5.299921402355064 correct 37 time  3.252124071121216<br />
Epoch  10  loss  4.633070379132276 correct 42 time  0.10783758163452148<br />
Epoch  20  loss  4.484028763570538 correct 43 time  0.10790717601776123<br />
Epoch  30  loss  4.381939633507319 correct 41 time  0.10693988800048829<br />
Epoch  40  loss  3.55199775309371 correct 44 time  0.10139615535736084<br />
Epoch  50  loss  3.666321488303418 correct 42 time  0.10133886337280273<br />
Epoch  60  loss  2.5159408974080084 correct 47 time  0.10256469249725342<br />
Epoch  70  loss  2.6942000922738973 correct 41 time  0.10297107696533203<br />
Epoch  80  loss  2.7493567612557714 correct 46 time  0.16726295948028563<br />
Epoch  90  loss  3.5412051509757445 correct 47 time  0.19188876152038575<br />
Epoch  100  loss  1.7110530550825804 correct 47 time  0.2315603256225586<br />
Epoch  110  loss  3.816455602325729 correct 47 time  0.22119007110595704<br />
Epoch  120  loss  1.560626995714669 correct 48 time  0.2453084707260132<br />
Epoch  130  loss  0.6958694705533601 correct 49 time  0.23494791984558105<br />
Epoch  140  loss  1.3792586380031646 correct 48 time  0.18592162132263185<br />
Epoch  150  loss  1.9961823924286348 correct 48 time  0.10482873916625976<br />
Epoch  160  loss  1.459271828746884 correct 49 time  0.10614306926727295<br />
Epoch  170  loss  0.768256455768002 correct 49 time  0.104990553855896<br />
Epoch  180  loss  1.4397678929470545 correct 48 time  0.10528011322021484<br />
Epoch  190  loss  0.8551625115819071 correct 48 time  0.10606110095977783<br />
Epoch  200  loss  0.5873575768405898 correct 49 time  0.10761582851409912<br />
Epoch  210  loss  0.9764165076788863 correct 49 time  0.10786755084991455<br />
Epoch  220  loss  1.1550316687087918 correct 50 time  0.11077086925506592<br />
Epoch  230  loss  0.6357216477953883 correct 49 time  0.11158950328826904<br />
Epoch  240  loss  0.446268446597001 correct 49 time  0.1964186668395996<br />
Epoch  250  loss  0.4282445518853339 correct 49 time  0.20899083614349365<br />
Epoch  260  loss  0.39841030855706394 correct 49 time  0.22504885196685792<br />
Epoch  270  loss  1.89305827315347 correct 49 time  0.23427858352661132<br />
Epoch  280  loss  1.4847880495724208 correct 49 time  0.23860018253326415<br />
Epoch  290  loss  1.7483724127962954 correct 50 time  0.23356435298919678<br />
Epoch  300  loss  0.7311644081368974 correct 49 time  0.160269832611084<br />
Epoch  310  loss  1.6198929079571516 correct 49 time  0.10110158920288086<br />
Epoch  320  loss  0.19577606655615115 correct 50 time  0.10245471000671387<br />
Epoch  330  loss  0.30497597904998053 correct 49 time  0.10677037239074708<br />
Epoch  340  loss  0.1488945394243528 correct 50 time  0.10085575580596924<br />
Epoch  350  loss  1.0446155222082072 correct 50 time  0.10111870765686035<br />
Epoch  360  loss  0.33994908532937795 correct 50 time  0.11056454181671142<br />
Epoch  370  loss  0.9174276548134385 correct 50 time  0.10149056911468506<br />
Epoch  380  loss  0.6898314325814304 correct 50 time  0.10336809158325196<br />
Epoch  390  loss  1.2468951868853557 correct 49 time  0.10119767189025879<br />
Epoch  400  loss  0.48188536513461844 correct 49 time  0.2052527904510498<br />
Epoch  410  loss  1.0153935721760465 correct 50 time  0.19841318130493163<br />
Epoch  420  loss  0.840696628930934 correct 49 time  0.25043938159942625<br />
Epoch  430  loss  0.06598848775445876 correct 50 time  0.2270939826965332<br />
Epoch  440  loss  0.30063297956555773 correct 50 time  0.2402663230895996<br />
Epoch  450  loss  0.9703573431293209 correct 50 time  0.23538503646850586<br />
Epoch  460  loss  0.6661011070476996 correct 50 time  0.12262003421783448<br />
Epoch  470  loss  0.39585895403806814 correct 50 time  0.10201468467712402<br />
Epoch  480  loss  1.003392655236698 correct 49 time  0.1097198486328125<br />
Epoch  490  loss  0.37205079769409777 correct 49 time  0.10947990417480469<br />
<br />
<br />
### Large Hidden run_fast_tensor GPU Hidden=200 Dataset=split<br />
!cd $DIR; PYTHONPATH=/content/$DIR python3.8 project/run_fast_tensor.py --BACKEND gpu --HIDDEN 200 --DATASET split --RATE 0.05<br />
Epoch  0  loss  7.672018706345448 correct 16 time  0.7546385765075684<br />
Epoch  10  loss  3.120106239277019 correct 40 time  2.51077778339386<br />
Epoch  20  loss  3.023036418101318 correct 48 time  2.507030963897705<br />
Epoch  30  loss  1.4730896164202205 correct 48 time  2.4943167924880982<br />
Epoch  40  loss  0.6862411629775431 correct 46 time  2.5012243032455443<br />
Epoch  50  loss  2.0195441750812253 correct 48 time  3.0699296236038207<br />
Epoch  60  loss  1.7180003662816783 correct 49 time  2.562162399291992<br />
Epoch  70  loss  0.6226589925558124 correct 49 time  2.561291217803955<br />
Epoch  80  loss  1.1221453292247836 correct 50 time  2.617898106575012<br />
Epoch  90  loss  0.6308059850016708 correct 48 time  2.565973329544067<br />
Epoch  100  loss  1.1400216116045256 correct 49 time  2.5963703870773314<br />
Epoch  110  loss  1.5250950569899047 correct 50 time  2.532428812980652<br />
Epoch  120  loss  0.3846224727096049 correct 49 time  2.482088541984558<br />
Epoch  130  loss  1.201998061121944 correct 50 time  2.494864511489868<br />
Epoch  140  loss  0.7495897539789586 correct 50 time  2.4980432271957396<br />
Epoch  150  loss  0.35612344708328963 correct 50 time  2.8335062503814696<br />
Epoch  160  loss  0.8020834336674838 correct 49 time  2.5989373207092283<br />
Epoch  170  loss  0.7941811149541363 correct 49 time  2.5467503309249877<br />
Epoch  180  loss  0.4979266653430334 correct 50 time  2.545016813278198<br />
Epoch  190  loss  0.7335151101406872 correct 50 time  2.5576317071914674<br />
Epoch  200  loss  1.0532166289281506 correct 50 time  2.635746192932129<br />
Epoch  210  loss  0.640124821697876 correct 50 time  2.5814236879348753<br />
Epoch  220  loss  1.2607227478335818 correct 50 time  2.5875159025192263<br />
Epoch  230  loss  0.516609018511502 correct 50 time  2.5738789558410646<br />
Epoch  240  loss  0.34307595406505514 correct 50 time  2.5425652503967284<br />
Epoch  250  loss  0.14337240967724066 correct 50 time  2.653263545036316<br />
Epoch  260  loss  0.5014974828803631 correct 49 time  2.6646080493927<br />
Epoch  270  loss  0.7214250226367768 correct 49 time  2.506300711631775<br />
Epoch  280  loss  0.8751404535080927 correct 50 time  2.5061163663864137<br />
Epoch  290  loss  0.5089333187920562 correct 49 time  2.5339018583297728<br />
Epoch  300  loss  0.6092950293826589 correct 50 time  2.5162214756011965<br />
Epoch  310  loss  0.70245269976386 correct 50 time  2.5498539686203<br />
Epoch  320  loss  0.3070731128424148 correct 50 time  2.6151599884033203<br />
Epoch  330  loss  0.09636372750546168 correct 49 time  2.59498610496521<br />
Epoch  340  loss  0.19968864002179373 correct 50 time  2.593751549720764<br />
Epoch  350  loss  0.3726042432305845 correct 50 time  2.667386364936829<br />
Epoch  360  loss  0.10466956795433578 correct 50 time  2.5729944467544557<br />
Epoch  370  loss  0.2339793401680499 correct 50 time  2.553938126564026<br />
Epoch  380  loss  2.5703038882845504 correct 44 time  2.5980618715286257<br />
Epoch  390  loss  0.2280586409293435 correct 50 time  2.5585195302963255<br />
Epoch  400  loss  0.5671930076026149 correct 50 time  2.4897576808929442<br />
Epoch  410  loss  0.3576762765045187 correct 50 time  2.4900566577911376<br />
Epoch  420  loss  0.37023753116021624 correct 50 time  2.5024922132492065<br />
Epoch  430  loss  0.1521876527786814 correct 50 time  2.4974443197250364<br />
Epoch  440  loss  0.2710985154760036 correct 50 time  2.529303860664368<br />
Epoch  450  loss  0.19153005988891392 correct 50 time  2.8950435638427736<br />
Epoch  460  loss  0.18724443612586553 correct 50 time  2.608188533782959<br />
Epoch  470  loss  0.1854931040303585 correct 50 time  2.5537073612213135<br />
Epoch  480  loss  0.17563606217824712 correct 50 time  2.5569568872451782<br />
Epoch  490  loss  0.09835595117320772 correct 50 time  2.5985013008117677<br />
<br />
<br />
### Large Hidden run_fast_tensor CPU Hidden=200 Dataset=split<br />
!cd $DIR; PYTHONPATH=/content/$DIR python3.8 project/run_fast_tensor.py --BACKEND cpu --HIDDEN 200 --DATASET split --RATE 0.05<br />
Epoch  0  loss  6.5480579221427435 correct 29 time  3.1765054941177366<br />
Epoch  10  loss  3.738544637746876 correct 46 time  0.41595361232757566<br />
Epoch  20  loss  2.247980848317383 correct 46 time  0.4503523349761963<br />
Epoch  30  loss  3.3970486201493064 correct 47 time  0.4471702575683594<br />
Epoch  40  loss  1.384087828071373 correct 47 time  0.22777869701385497<br />
Epoch  50  loss  2.809924028140388 correct 47 time  0.21597435474395751<br />
Epoch  60  loss  2.11243360994052 correct 50 time  0.21575257778167725<br />
Epoch  70  loss  1.1835639345829252 correct 47 time  0.21745364665985106<br />
Epoch  80  loss  1.4378721544289046 correct 49 time  0.2795127868652344<br />
Epoch  90  loss  1.6968182859883922 correct 48 time  0.4195310354232788<br />
Epoch  100  loss  1.3118438200212739 correct 49 time  0.4462618589401245<br />
Epoch  110  loss  0.9898537590459814 correct 50 time  0.4423495292663574<br />
Epoch  120  loss  1.8777358291297506 correct 50 time  0.21483197212219238<br />
Epoch  130  loss  1.0696052448359599 correct 50 time  0.21565570831298828<br />
Epoch  140  loss  1.4132078202434133 correct 49 time  0.2139119863510132<br />
Epoch  150  loss  0.7463857314491332 correct 50 time  0.21489803791046141<br />
Epoch  160  loss  0.2399650111145555 correct 49 time  0.29024245738983157<br />
Epoch  170  loss  1.0254982654542548 correct 50 time  0.4203095674514771<br />
Epoch  180  loss  1.053899790500487 correct 50 time  0.45527029037475586<br />
Epoch  190  loss  0.8131485241536422 correct 50 time  0.42124860286712645<br />
Epoch  200  loss  0.48846499710959324 correct 50 time  0.22673890590667725<br />
Epoch  210  loss  0.7086320387077175 correct 50 time  0.21253693103790283<br />
Epoch  220  loss  0.6490658747652674 correct 50 time  0.21997442245483398<br />
Epoch  230  loss  0.09261685953906705 correct 50 time  0.21348235607147217<br />
Epoch  240  loss  0.4690282558317531 correct 50 time  0.3146009922027588<br />
Epoch  250  loss  0.4476691508843679 correct 50 time  0.4275991201400757<br />
Epoch  260  loss  0.45478472360954914 correct 50 time  0.456441855430603<br />
Epoch  270  loss  0.36259053388261075 correct 50 time  0.3913256645202637<br />
Epoch  280  loss  0.2230873792698109 correct 50 time  0.21508591175079345<br />
Epoch  290  loss  0.40502233290531103 correct 50 time  0.21447925567626952<br />
Epoch  300  loss  0.06508974762466976 correct 50 time  0.2167125463485718<br />
Epoch  310  loss  0.5241311681453207 correct 50 time  0.22002770900726318<br />
Epoch  320  loss  0.26129808260250204 correct 50 time  0.3410566568374634<br />
Epoch  330  loss  0.6062611617067579 correct 50 time  0.42892520427703856<br />
Epoch  340  loss  0.5486401635296758 correct 50 time  0.4566131353378296<br />
Epoch  350  loss  0.6371661373145746 correct 50 time  0.36784493923187256<br />
Epoch  360  loss  0.2795201353474401 correct 50 time  0.24833009243011475<br />
Epoch  370  loss  0.08467186027234999 correct 50 time  0.21498641967773438<br />
Epoch  380  loss  0.03375563938762882 correct 50 time  0.21549503803253173<br />
Epoch  390  loss  0.0961345667017856 correct 50 time  0.2139117956161499<br />
Epoch  400  loss  0.2780311199837733 correct 50 time  0.38727555274963377<br />
Epoch  410  loss  0.3952924043807351 correct 50 time  0.44152164459228516<br />
Epoch  420  loss  0.13152886751123108 correct 50 time  0.4660893440246582<br />
Epoch  430  loss  0.6277964644178571 correct 50 time  0.30511677265167236<br />
Epoch  440  loss  0.19068692347982172 correct 50 time  0.21708106994628906<br />
Epoch  450  loss  0.23051874138404482 correct 50 time  0.21860806941986083<br />
Epoch  460  loss  0.0932148280558422 correct 50 time  0.21703431606292725<br />
Epoch  470  loss  0.14648269213384948 correct 50 time  0.22383646965026854<br />
Epoch  480  loss  0.4050733259290595 correct 50 time  0.4160802125930786<br />
Epoch  490  loss  0.14448313768268398 correct 50 time  0.44319198131561277<br />

